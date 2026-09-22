"""
Attack: Seme+HIQPSO-RD — BERT on IMDB  (multi-worker version)
================================================================
Search space  : HowNet sememe-based substitutes
Search method : HIQPSO-RD — Historical Information-guided QPSO
                with Random Drift local attractor  (proposed method)

Implements ALL equations from Section III of the paper:
  Eq.  6  — single-mutation initialisation
  Eq.  7  — mutation update per iteration
  Eq.  8  — binary substitute-identify function I(·)
  Eq.  9  — HIQPSO-RD position update  (core equation)
  Eq. 10  — historical mean best  HC_t
  Eq. 11  — current mean best  C_t  (over binary-encoded personal bests)
  Eq. 12  — local attractor  p_{i,t}
  Eq. 13  — sigmoid mutation probability  PM_X
  Eq. 14  — two-stage diversity control  (α > 0.75 → pbest; else → gbest)

Hyperparameters (Section IV-D):
  T  = 20,  M = 60,  α: 1.0→0.5 (linear),  β = 0.75,  m1 = 0.1

This version processes multiple IMDB samples CONCURRENTLY using separate
worker processes (num_workers), each sharing the same GPU. This keeps the
GPU busier than running samples strictly one-after-another.

Resuming: the CSV file itself is the checkpoint. Re-running this script
automatically skips any sample index already present in the CSV — no
separate checkpoint file needed.

Prerequisite:
    pip install OpenAttack --break-system-packages
    python -c "import OpenAttack; OpenAttack.download('AttackAssist.HownetSubstituteDict')"

Usage:
    python attack_hiqpso_bert_imdb.py                  # default: 4 workers
    python attack_hiqpso_bert_imdb.py --num_workers 2   # fewer workers if VRAM is tight
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import math
import pickle
import random
import time
import multiprocessing as mp

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import OpenAttack
from filelock import FileLock

import datasets.utils.info_utils
import datasets.builder
datasets.utils.info_utils.verify_splits = lambda *args, **kwargs: None
datasets.builder.verify_splits = lambda *args, **kwargs: None
try:
    import datasets.utils.file_utils
    datasets.utils.file_utils.verify_splits = lambda *args, **kwargs: None
except Exception:
    pass
from datasets import load_dataset


RESULTS_FILE = "results_hiqpso_bert_imdb.csv"
LOCK_FILE    = RESULTS_FILE + ".lock"
CSV_COLUMNS  = [
    "sample_idx",
    "original_text", "perturbed_text",
    "original_score", "perturbed_score",
    "original_output", "perturbed_output",
    "ground_truth_output", "num_queries", "result_type",
]


# ======================================================================= #
#  VICTIM WRAPPER                                                          #
# ======================================================================= #

class BertVictim(OpenAttack.Classifier):
    def __init__(self, model, tokenizer, device):
        self._model      = model
        self._tokenizer  = tokenizer
        self._device     = device
        self.query_count = 0

    def reset_queries(self):
        self.query_count = 0

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        self.query_count += len(input_)
        enc = self._tokenizer(
            input_,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()


# ======================================================================= #
#  HOWNET SUBSTITUTE HELPER                                                #
# ======================================================================= #

_HOWNET_POS_TAGS = ["noun", "verb", "adj", "adv", "other"]

def load_hownet_substitute_dict():
    pkl_path = OpenAttack.DataManager.load("AttackAssist.HownetSubstituteDict")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_sememe_substitutes(word: str, hownet_dict: dict) -> list:
    if word not in hownet_dict:
        return []
    seen, result = {word}, []
    for pos_tag in _HOWNET_POS_TAGS:
        for candidate in hownet_dict[word].get(pos_tag, []):
            if isinstance(candidate, str) and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def build_search_space(tokens: list, hownet_dict: dict) -> list:
    return [[tok] + get_sememe_substitutes(tok, hownet_dict) for tok in tokens]


# ======================================================================= #
#  BATCHED FITNESS                                                         #
# ======================================================================= #

def fitness_batch(token_lists: list, target_label: int,
                  victim: BertVictim, chunk: int = 32) -> list:
    scores = []
    for i in range(0, len(token_lists), chunk):
        texts = [" ".join(t) for t in token_lists[i:i + chunk]]
        probs = victim.get_prob(texts)
        scores.extend(float(p[target_label]) for p in probs)
    return scores


# ======================================================================= #
#  BINARY ENCODING  I(·)   (Equation 8)                                   #
# ======================================================================= #

def encode(particle: list, original: list) -> list:
    return [0 if p == o else 1 for p, o in zip(particle, original)]


# ======================================================================= #
#  INITIALISATION  (Equation 6)                                            #
# ======================================================================= #

def single_mutation_init(S0, search_space, M, target_label, victim):
    D = len(S0)

    all_cands, cand_pos = [], []
    for j in range(D):
        for sub in search_space[j][1:]:
            cand    = list(S0)
            cand[j] = sub
            all_cands.append(cand)
            cand_pos.append(j)

    base_score = fitness_batch([S0], target_label, victim)[0]
    all_scores = fitness_batch(all_cands, target_label, victim) if all_cands else []

    WM0   = list(S0)
    gains = [0.0] * D

    idx = 0
    for j in range(D):
        best_sub, best_score = S0[j], base_score
        for sub in search_space[j][1:]:
            if all_scores[idx] > best_score:
                best_score = all_scores[idx]
                best_sub   = sub
            idx += 1
        WM0[j]   = best_sub
        gains[j] = max(0.0, best_score - base_score)

    gain_sum = sum(gains) + 1e-9
    PM0      = [g / gain_sum for g in gains]

    swarm = []
    for _ in range(M):
        particle = list(S0)
        for j in range(D):
            if random.random() < PM0[j]:
                particle[j] = WM0[j]
        swarm.append(particle)

    return swarm


# ======================================================================= #
#  MUTATION UPDATE  (Equation 7)                                           #
# ======================================================================= #

def mutation_update(particle, S0, search_space, target_label, victim):
    D = len(S0)

    all_cands, cand_meta = [], []
    for j in range(D):
        options = [S0[j]] + search_space[j][1:]
        for sub in options:
            if sub == particle[j]:
                continue
            cand    = list(particle)
            cand[j] = sub
            all_cands.append(cand)
            cand_meta.append((j, sub))

    base_score = fitness_batch([particle], target_label, victim)[0]
    all_scores = fitness_batch(all_cands, target_label, victim) if all_cands else []

    best_sub   = list(particle)
    best_score = [base_score] * D
    gains      = [0.0] * D

    for k, (j, sub) in enumerate(cand_meta):
        if all_scores[k] > best_score[j]:
            best_score[j] = all_scores[k]
            best_sub[j]   = sub
            gains[j]      = all_scores[k] - base_score

    gain_sum     = sum(gains) + 1e-9
    PM           = [max(0.0, g) / gain_sum for g in gains]
    new_particle = list(particle)
    for j in range(D):
        if random.random() < PM[j]:
            new_particle[j] = best_sub[j]
    return new_particle


# ======================================================================= #
#  SIGMOID                                                                 #
# ======================================================================= #

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


# ======================================================================= #
#  HIQPSO-RD POSITION UPDATE  (Equations 9-14)                            #
# ======================================================================= #

def hiqpso_rd_update(swarm, personal_best, global_best, prev_mean_best,
                     S0, search_space, alpha_t, beta, m1):
    M = len(swarm)
    D = len(S0)

    I_pbest = [encode(pb, S0) for pb in personal_best]
    I_gbest = encode(global_best, S0)

    C_t = [sum(I_pbest[i][j] for i in range(M)) / M for j in range(D)]

    HC_t = [
        m1 * random.random() * prev_mean_best[j]
        + (1.0 - m1 * random.random()) * C_t[j]
        for j in range(D)
    ]

    new_swarm = []
    for i in range(M):
        I_xi = encode(swarm[i], S0)

        phi  = [random.random() for _ in range(D)]
        p_it = [
            phi[j] * I_pbest[i][j] + (1.0 - phi[j]) * I_gbest[j]
            for j in range(D)
        ]

        p_drift = [
            p_it[j] + beta * abs(p_it[j] - I_xi[j]) * random.gauss(0.0, 1.0)
            for j in range(D)
        ]

        sign  = [1 if random.random() > 0.5 else -1 for _ in range(D)]
        u     = [max(1e-9, random.random()) for _ in range(D)]
        X_new = [
            p_drift[j]
            + sign[j] * alpha_t * abs(HC_t[j] - I_xi[j]) * math.log(1.0 / u[j])
            for j in range(D)
        ]

        PM_X = [sigmoid(x) for x in X_new]

        new_particle = list(swarm[i])
        for j in range(D):
            if random.random() > PM_X[j]:
                if alpha_t > 0.75:
                    new_particle[j] = personal_best[i][j]
                else:
                    new_particle[j] = global_best[j]

        for j in range(D):
            if new_particle[j] not in search_space[j]:
                new_particle[j] = S0[j]

        new_swarm.append(new_particle)

    return new_swarm, C_t


# ======================================================================= #
#  FULL HIQPSO-RD ATTACK FOR ONE SENTENCE  (Algorithm 1)                  #
# ======================================================================= #

def attack_sentence(S0_tokens, true_label, victim, hownet_dict,
                    T=20, M=60, beta=0.75, m1=0.1):
    target_label = 1 - true_label
    D = len(S0_tokens)
    if D == 0:
        return None

    search_space = build_search_space(S0_tokens, hownet_dict)
    swarm        = single_mutation_init(S0_tokens, search_space, M,
                                        target_label, victim)

    init_scores = fitness_batch(swarm, target_label, victim)
    for p, s in zip(swarm, init_scores):
        if s > 0.5:
            return p

    personal_best  = [list(p) for p in swarm]
    pb_fitness     = list(init_scores)
    best_idx       = int(np.argmax(pb_fitness))
    global_best    = list(personal_best[best_idx])

    I_pbest_init   = [encode(pb, S0_tokens) for pb in personal_best]
    prev_mean_best = [sum(I_pbest_init[i][j] for i in range(M)) / M
                      for j in range(D)]

    for t in range(1, T + 1):
        alpha_t = 1.0 - 0.5 * (t - 1) / max(1, T - 1)

        new_swarm = [
            mutation_update(swarm[i], S0_tokens, search_space, target_label, victim)
            for i in range(M)
        ]

        mut_scores = fitness_batch(new_swarm, target_label, victim)
        for p, s in zip(new_swarm, mut_scores):
            if s > 0.5:
                return p

        swarm = new_swarm
        for i in range(M):
            if mut_scores[i] > pb_fitness[i]:
                personal_best[i] = list(swarm[i])
                pb_fitness[i]    = mut_scores[i]
                if mut_scores[i] > pb_fitness[best_idx]:
                    best_idx    = i
                    global_best = list(swarm[i])

        swarm, prev_mean_best = hiqpso_rd_update(
            swarm, personal_best, global_best, prev_mean_best,
            S0_tokens, search_space, alpha_t, beta, m1
        )

        pos_scores = fitness_batch(swarm, target_label, victim)
        for p, s in zip(swarm, pos_scores):
            if s > 0.5:
                return p

        for i in range(M):
            if pos_scores[i] > pb_fitness[i]:
                personal_best[i] = list(swarm[i])
                pb_fitness[i]    = pos_scores[i]
                if pos_scores[i] > pb_fitness[best_idx]:
                    best_idx    = i
                    global_best = list(swarm[i])

    return None


# ======================================================================= #
#  CSV HELPERS (process-safe via filelock)                                #
# ======================================================================= #

def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def append_row_safe(path, lock_path, row: dict):
    with FileLock(lock_path):
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

def load_done_indices(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["sample_idx"]))
            except (KeyError, ValueError):
                continue
    return done


# ======================================================================= #
#  WORKER PROCESS                                                          #
# ======================================================================= #

def worker_process(worker_id, sample_indices, samples, model_path,
                   hownet_pkl_path, device_str, results_file, lock_file):
    torch.manual_seed(worker_id)
    random.seed(worker_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device_str)
    model.eval()
    victim = BertVictim(model, tokenizer, device_str)

    with open(hownet_pkl_path, "rb") as f:
        hownet_dict = pickle.load(f)

    print(f"[Worker {worker_id}] Ready — {len(sample_indices)} samples assigned")

    for count, idx in enumerate(sample_indices):
        text, label = samples[idx]
        tokens = text.split()

        victim.reset_queries()
        orig_prob  = victim.get_prob([text])[0]
        orig_pred  = int(orig_prob.argmax())
        orig_score = float(orig_prob[orig_pred])

        if orig_pred != label:
            row = {
                "sample_idx": idx,
                "original_text": text, "perturbed_text": text,
                "original_score": round(orig_score, 6),
                "perturbed_score": round(orig_score, 6),
                "original_output": orig_pred, "perturbed_output": orig_pred,
                "ground_truth_output": label,
                "num_queries": victim.query_count,
                "result_type": "Skipped",
            }
        else:
            adv_tokens = attack_sentence(
                tokens, label, victim, hownet_dict,
                T=20, M=60, beta=0.75, m1=0.1
            )
            if adv_tokens is not None:
                adv_text  = " ".join(adv_tokens)
                adv_prob  = victim.get_prob([adv_text])[0]
                adv_pred  = int(adv_prob.argmax())
                adv_score = float(adv_prob[adv_pred])
                result_type = "Successful" if adv_pred != label else "Failed"
                row = {
                    "sample_idx": idx,
                    "original_text": text, "perturbed_text": adv_text,
                    "original_score": round(orig_score, 6),
                    "perturbed_score": round(adv_score, 6),
                    "original_output": orig_pred, "perturbed_output": adv_pred,
                    "ground_truth_output": label,
                    "num_queries": victim.query_count,
                    "result_type": result_type,
                }
            else:
                row = {
                    "sample_idx": idx,
                    "original_text": text, "perturbed_text": text,
                    "original_score": round(orig_score, 6),
                    "perturbed_score": round(orig_score, 6),
                    "original_output": orig_pred, "perturbed_output": orig_pred,
                    "ground_truth_output": label,
                    "num_queries": victim.query_count,
                    "result_type": "Failed",
                }

        append_row_safe(results_file, lock_file, row)

        if (count + 1) % 5 == 0:
            print(f"[Worker {worker_id}] {count + 1}/{len(sample_indices)} done "
                  f"(global idx {idx})")

    print(f"[Worker {worker_id}] FINISHED")


# ======================================================================= #
#  MAIN                                                                    #
# ======================================================================= #

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel worker processes sharing the GPU. "
                             "Reduce if you hit CUDA OOM (try 2).")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[GPU] Using: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.device_count()} device(s) visible)")
        print(f"[INFO] Spawning {args.num_workers} worker(s) sharing this GPU")
    else:
        print("[CPU] No CUDA device found — running on CPU. "
              "Multi-worker still helps via CPU parallelism.")

    model_path = os.path.abspath("bert-base-uncased-imdb")

    print("[INFO] Resolving HowNet substitute dictionary path ...")
    hownet_pkl_path = OpenAttack.DataManager.load("AttackAssist.HownetSubstituteDict")

    raw_dataset = load_dataset("imdb", split="test")
    samples = []
    for sample in raw_dataset:
        truncated_text = " ".join(sample["text"].split()[:50])
        samples.append((truncated_text, sample["label"]))
    samples = samples[:args.num_samples]

    init_csv(RESULTS_FILE)
    done_indices = load_done_indices(RESULTS_FILE)
    if done_indices:
        print(f"[RESUME] {len(done_indices)} samples already in CSV — skipping those")

    remaining_indices = [i for i in range(len(samples)) if i not in done_indices]
    print(f"[INFO] {len(remaining_indices)} samples remaining to process")

    if not remaining_indices:
        print("[DONE] All samples already processed.")
    else:
        worker_chunks = [[] for _ in range(args.num_workers)]
        for k, idx in enumerate(remaining_indices):
            worker_chunks[k % args.num_workers].append(idx)

        start_time = time.time()
        procs = []
        for wid in range(args.num_workers):
            if not worker_chunks[wid]:
                continue
            p = mp.Process(
                target=worker_process,
                args=(wid, worker_chunks[wid], samples, model_path,
                     hownet_pkl_path, device, RESULTS_FILE, LOCK_FILE),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        elapsed = time.time() - start_time
        print(f"\n[INFO] All workers finished in {elapsed / 3600:.2f} hours")

    # ------------------------------------------------------------------ #
    #  FINAL METRICS — recomputed from the full CSV                      #
    # ------------------------------------------------------------------ #
    rows = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    y_true      = [int(r["ground_truth_output"]) for r in rows]
    y_pred_orig = [int(r["original_output"]) for r in rows]
    y_pred_adv  = [int(r["perturbed_output"]) for r in rows]
    n_success   = sum(1 for r in rows if r["result_type"] == "Successful")

    orig_acc = accuracy_score(y_true, y_pred_orig)
    orig_p, orig_r, orig_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_orig, average="weighted", zero_division=0
    )
    adv_acc = accuracy_score(y_true, y_pred_adv)
    adv_p, adv_r, adv_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_adv, average="weighted", zero_division=0
    )

    print("\n" + "=" * 58)
    print("   Seme+HIQPSO-RD — BERT / IMDB   METRICS REPORT")
    print("=" * 58)
    print(f"Baseline accuracy  (before attack): {orig_acc * 100:.2f}%")
    print(f"  Precision: {orig_p:.4f}  Recall: {orig_r:.4f}  F1: {orig_f1:.4f}")
    print(f"Adversarial accuracy (after attack): {adv_acc * 100:.2f}%")
    print(f"  Precision: {adv_p:.4f}  Recall: {adv_r:.4f}  F1: {adv_f1:.4f}")
    print(f"Successful attacks: {n_success} / {len(rows)}")
    print(f"Results saved to:   {RESULTS_FILE}")
    print("=" * 58 + "\n")