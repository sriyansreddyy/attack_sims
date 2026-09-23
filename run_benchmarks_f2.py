"""
run_benchmarks.py
=================
Main benchmark script.  Runs F2Attack against the fine-tuned BERT/IMDB model,
exactly matching the paper's evaluation protocol (Section V):

  - 500 examples from test set (randomly selected)
  - Query budget: 100 (paper default)
  - Synonym set size k=5
  - Annealing rate alpha=0.9
  - Metrics: ASR (%), Semantic Similarity

Usage:
    python run_benchmarks.py \
        --glove path/to/glove.6B.200d.txt \
        --model_dir ./bert-base-uncased-imdb \
        --fattn_ckpt ./fattn_checkpoint.pt \
        [--n_samples 500] \
        [--query_budget 100] \
        [--output results.json]

Output:
    results.json    <- per-sample results
    benchmark_table.txt  <- summary table for your report
"""

import argparse, json, random, time, os
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

from f2attack_core import (
    Fattn, EmbeddingProvider, SimScore, SynonymDict,
    two_factors_word_scoring, f2attack
)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--glove",        required=True)
parser.add_argument("--model_dir",    default="./bert-base-uncased-imdb")
parser.add_argument("--fattn_ckpt",   default="./fattn_checkpoint.pt")
parser.add_argument("--n_samples",    type=int,   default=500)
parser.add_argument("--query_budget", type=int,   default=100)
parser.add_argument("--k_synonyms",   type=int,   default=5)
parser.add_argument("--anneal_rate",  type=float, default=0.9)
parser.add_argument("--sim_batch",    type=int,   default=256,  help="Sentence encoder batch size (increase for more VRAM)")
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--output",       default="results.json")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load target model (BERT fine-tuned on IMDB)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading BERT target model …")
tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
bert_model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
bert_model.eval()

@torch.no_grad()
def target_model_fn(text: str) -> int:
    """Query function exposed to the attacker (decision output only)."""
    enc  = tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=512, padding=True).to(device)
    out  = bert_model(**enc)
    return int(out.logits.argmax(dim=-1).item())

@torch.no_grad()
def target_model_accuracy(texts, labels):
    """Batch evaluation for baseline accuracy."""
    correct = 0
    for text, label in zip(texts, labels):
        if target_model_fn(text) == label:
            correct += 1
    return correct / len(labels) * 100

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load IMDB test set, select 500 correctly classified examples
#    (paper: attack only on correctly classified examples)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading IMDB test data …")
raw_test = load_dataset("imdb")["test"]

# Randomly sample candidates
all_indices = list(range(len(raw_test)))
random.shuffle(all_indices)

print(f"Selecting {args.n_samples} correctly classified examples …")
selected_texts, selected_labels = [], []

for idx in tqdm(all_indices):
    if len(selected_texts) >= args.n_samples:
        break
    text  = raw_test[idx]["text"]
    label = raw_test[idx]["label"]
    # Only include correctly classified (paper protocol)
    if target_model_fn(text) == label:
        selected_texts.append(text)
        selected_labels.append(label)

N_acc = len(selected_texts)
print(f"Found {N_acc} correctly classified samples (N_acc)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Baseline accuracy
# ─────────────────────────────────────────────────────────────────────────────
# All selected examples are correctly classified by construction
baseline_acc = 100.0   # N_acc / N_acc
print(f"\nBaseline Accuracy on selected samples: {baseline_acc:.2f}%")
print(f"(Full test set accuracy recorded in baseline_accuracy.txt)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load F2Attack components
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading F2Attack components …")

# 4a. Embedding provider
emb_provider = EmbeddingProvider(args.glove, args.model_dir, device=str(device))

# 4b. Fattn
fattn = Fattn(glove_dim=200, bert_dim=768, hidden_dim=150, num_classes=2).to(device)
if os.path.exists(args.fattn_ckpt):
    fattn.load_state_dict(torch.load(args.fattn_ckpt, map_location=device))
    print(f"  Loaded Fattn from {args.fattn_ckpt}")
else:
    print(f"  WARNING: {args.fattn_ckpt} not found. Using random Fattn weights.")
    print("  Run train_fattn.py first for proper results.")
fattn.eval()

# 4c. SimScore
print("  Loading sentence encoder …")
sim_scorer = SimScore(device=str(device), batch_size=args.sim_batch)

# 4d. Synonym dictionary
syn_dict = SynonymDict(k=args.k_synonyms)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run F2Attack on all 500 samples
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning F2Attack (budget={args.query_budget}, k={args.k_synonyms}, α={args.anneal_rate}) …")
print("=" * 60)

results       = []
n_success     = 0
total_sim     = 0.0
total_queries = 0
start_time    = time.time()

for i, (text, label) in enumerate(tqdm(zip(selected_texts, selected_labels), total=N_acc)):
    result = f2attack(
        text          = text,
        true_label    = label,
        target_model_fn = target_model_fn,
        fattn         = fattn,
        emb_provider  = emb_provider,
        sim_scorer    = sim_scorer,
        syn_dict      = syn_dict,
        query_budget  = args.query_budget,
        T_init        = 1.0,
        anneal_rate   = args.anneal_rate,
    )

    if result["success"]:
        n_success     += 1
        total_sim     += result["sim"]
        total_queries += result["queries"]

    results.append({
        "idx"      : i,
        "label"    : label,
        "text"     : text[:200] + "…" if len(text) > 200 else text,
        "adv_text" : result["adv_text"][:200] + "…" if len(result["adv_text"]) > 200 else result["adv_text"],
        "success"  : result["success"],
        "sim"      : round(result["sim"], 4),
        "queries"  : result["queries"],
    })

    # Live progress every 50 samples
    if (i + 1) % 50 == 0:
        curr_asr = n_success / (i + 1) * 100
        curr_sim = total_sim / max(n_success, 1)
        elapsed  = time.time() - start_time
        print(f"  [{i+1}/{N_acc}] ASR={curr_asr:.1f}% | Sim={curr_sim:.3f} | "
              f"Elapsed={elapsed/60:.1f}min")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Compute final metrics  (Eq. 12-14)
# ─────────────────────────────────────────────────────────────────────────────

# ASR = N_as / N_acc  (Eq. 13)
ASR = n_success / N_acc * 100

# Average semantic similarity over successful attacks
avg_sim = total_sim / max(n_success, 1)

# Average queries (efficiency metric, Section V-D)
avg_queries = total_queries / max(n_success, 1)

elapsed_total = time.time() - start_time

# ─────────────────────────────────────────────────────────────────────────────
# 7. Print benchmark table
# ─────────────────────────────────────────────────────────────────────────────

table = f"""
╔══════════════════════════════════════════════════════════════════╗
║          F2Attack Benchmark Results — BERT / IMDB               ║
╠══════════════════════════════════════════════════════════════════╣
║  Setting                                                        ║
║    Query budget (Q)    : {args.query_budget:<5}                                 ║
║    Synonym set size (k): {args.k_synonyms:<5}                                 ║
║    Annealing rate (α)  : {args.anneal_rate:<5}                                 ║
║    N samples (N_acc)   : {N_acc:<5}                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Results                                                        ║
║    Baseline Accuracy   : {baseline_acc:.2f}%                              ║
║    Attack Success Rate : {ASR:.3f}%   (paper: ~69.5%)             ║
║    Semantic Similarity : {avg_sim:.3f}    (paper: ~0.554)            ║
║    Avg Queries Used    : {avg_queries:.1f}                               ║
║    Total Time          : {elapsed_total/60:.1f} min                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Paper Reference (Table IV, BERT on FAS dataset, Q=100)        ║
║    ASR     : 69.546%                                            ║
║    Sim     : 0.554                                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

print(table)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save results
# ─────────────────────────────────────────────────────────────────────────────
with open(args.output, "w") as f:
    json.dump({
        "config" : vars(args),
        "summary": {
            "N_acc"      : N_acc,
            "N_success"  : n_success,
            "ASR_pct"    : round(ASR, 3),
            "avg_sim"    : round(avg_sim, 4),
            "avg_queries": round(avg_queries, 1),
            "elapsed_min": round(elapsed_total / 60, 1),
        },
        "samples": results,
    }, f, indent=2)

with open("benchmark_table.txt", "w") as f:
    f.write(table)

print(f"Results saved to: {args.output}")
print(f"Table saved to:   benchmark_table.txt")