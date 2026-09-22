"""
Attack: Syn+Greedy — BERT on IMDB
===================================
Search space  : WordNet synonyms  (same as the paper's "Syn+Greedy" baseline)
Search method : Greedy word-importance ranking (GreedyWordSwapWIR)
Constraints   : RepeatModification, StopwordModification,
                MaxWordsPerturbed (20%), WordEmbeddingDistance

Paper reference (Section II-A / Table I):
  "Syn+Greedy selects the input words' synonyms from WordNet in order
   to form the search space and applies the greedy search method to
   find a satisfying adversarial example."  [Ren et al., ACL 2019]

Follows the exact same structure as the user's attack_ga.py template.

Usage:
    python attack_syn_greedy_bert_imdb.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import textattack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapWordNet          # WordNet synonyms
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.attack import Attack


def extract_val(val):
    return val.item() if hasattr(val, "item") else val


if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  MODEL SETUP                                                         #
    # ------------------------------------------------------------------ #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[GPU] Using: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.device_count()} device(s) visible)")
    else:
        print("[CPU] No CUDA device found — running on CPU (expect slow runtime)")

    local_model_path = os.path.abspath("bert-base-uncased-imdb")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path, local_files_only=True
    ).to(device)
    model.eval()

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    # ------------------------------------------------------------------ #
    #  DATASET SETUP — IMDB test split, first 50 words per review         #
    # ------------------------------------------------------------------ #
    raw_dataset = load_dataset("imdb", split="test")

    dataset_elements = []
    for sample in raw_dataset:
        truncated_text = " ".join(sample["text"].split()[:50])
        dataset_elements.append((truncated_text, sample["label"]))

    textattack_dataset = textattack.datasets.Dataset(dataset_elements)

    # ------------------------------------------------------------------ #
    #  ATTACK GRAPH                                                        #
    # ------------------------------------------------------------------ #
    goal_function  = UntargetedClassification(model_wrapper)

    # WordNet synonyms — the "Syn" part of Syn+Greedy
    transformation = WordSwapWordNet()

    constraints = [
        RepeatModification(),
        StopwordModification(),
        # Same 20% budget as the paper's other methods
        MaxWordsPerturbed(max_percent=0.2, compare_against_original=True),
        # Embedding distance keeps substitutes semantically close
        WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=False),
    ]

    # Greedy search ranked by word importance (WIR = Word Importance Ranking)
    # This is the standard greedy baseline used in the NLP attack literature
    search_method = GreedyWordSwapWIR(wir_method="delete")

    attack = Attack(goal_function, constraints, transformation, search_method)

    # ------------------------------------------------------------------ #
    #  ATTACK ARGS                                                         #
    # ------------------------------------------------------------------ #
    attack_args = textattack.AttackArgs(
        num_examples=1000,
        log_to_csv="results_syn_greedy_bert_imdb.csv",
        checkpoint_interval=100,
        checkpoint_dir="checkpoints_syn_greedy_bert_imdb",
        disable_stdout=False,
        parallel=True,
        num_workers_per_device=3,
    )

    print("[START] Syn+Greedy Attack on BERT/IMDB ...")
    attacker = textattack.Attacker(attack, textattack_dataset, attack_args)
    results  = attacker.attack_dataset()

    # ------------------------------------------------------------------ #
    #  METRICS                                                             #
    # ------------------------------------------------------------------ #
    y_true, y_pred_orig, y_pred_adv = [], [], []

    for result in results:
        y_true.append(extract_val(result.original_result.ground_truth_output))
        y_pred_orig.append(extract_val(result.original_result.output))
        if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
            y_pred_adv.append(extract_val(result.perturbed_result.output))
        else:
            y_pred_adv.append(extract_val(result.original_result.output))

    orig_acc = accuracy_score(y_true, y_pred_orig)
    orig_p, orig_r, orig_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_orig, average="weighted", zero_division=0
    )
    adv_acc = accuracy_score(y_true, y_pred_adv)
    adv_p, adv_r, adv_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_adv, average="weighted", zero_division=0
    )

    print("\n" + "=" * 54)
    print("      Syn+Greedy — BERT / IMDB   METRICS REPORT")
    print("=" * 54)
    print(f"Baseline accuracy  (before attack): {orig_acc * 100:.2f}%")
    print(f"  Precision: {orig_p:.4f}  Recall: {orig_r:.4f}  F1: {orig_f1:.4f}")
    print(f"Adversarial accuracy (after attack): {adv_acc * 100:.2f}%")
    print(f"  Precision: {adv_p:.4f}  Recall: {adv_r:.4f}  F1: {adv_f1:.4f}")
    print("=" * 54 + "\n")