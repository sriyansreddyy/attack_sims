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
from datasets import load_dataset

# Import core components to rebuild the attack graph natively
from textattack.search_methods import AlzantotGeneticAlgorithm
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.grammaticality.language_models import GPT2
from textattack.goal_functions import UntargetedClassification
from textattack.attack import Attack

# Define this outside the main block so worker processes can reference it if needed
def extract_val(val):
    return val.item() if hasattr(val, 'item') else val

# THE FIX: Wrap the execution pipeline in the main entry point guard
if __name__ == '__main__':
    
    # --- MODEL SETUP ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_path = os.path.abspath("bert-base-uncased-imdb")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True).to(device)

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    # --- DATASET SETUP ---
    raw_dataset = load_dataset("imdb", split="test")

    dataset_elements = []
    for sample in raw_dataset:
        truncated_text = " ".join(sample["text"].split()[:50])
        dataset_elements.append((truncated_text, sample["label"]))

    textattack_dataset = textattack.datasets.Dataset(dataset_elements)

    # --- ATTACK GRAPH CONSTRUCTION ---
    goal_function = UntargetedClassification(model_wrapper)
    transformation = WordSwapEmbedding(max_candidates=8)

    constraints = [
        RepeatModification(),
        StopwordModification(),
        MaxWordsPerturbed(max_percent=0.2, compare_against_original=True),
        WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=False),
        GPT2(max_log_prob_diff=2.0, compare_against_original=False) 
    ]

    search_method = AlzantotGeneticAlgorithm(
        pop_size=60, 
        max_iters=20, 
        post_crossover_check=False
    )

    attack = Attack(goal_function, constraints, transformation, search_method)

    # Parallel arguments now safely run inside the guarded block
    attack_args = textattack.AttackArgs(
        num_examples=1000,
        log_to_csv="results_ga.csv",
        checkpoint_interval=100,
        checkpoint_dir="checkpoints_ga",
        disable_stdout=False,
        parallel=True,
        num_workers_per_device=4
    )

    print("[START] Native PyTorch Genetic Algorithm Attack running...")
    attacker = textattack.Attacker(attack, textattack_dataset, attack_args)
    results = attacker.attack_dataset()

    # --- METRICS ENGINE ---
    y_true, y_pred_orig, y_pred_adv = [], [], []

    for result in results:
        y_true.append(extract_val(result.original_result.ground_truth_output))
        y_pred_orig.append(extract_val(result.original_result.output))
        if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
            y_pred_adv.append(extract_val(result.perturbed_result.output))
        else:
            y_pred_adv.append(extract_val(result.original_result.output))

    orig_acc = accuracy_score(y_true, y_pred_orig)
    orig_p, orig_r, orig_f1, _ = precision_recall_fscore_support(y_true, y_pred_orig, average='weighted', zero_division=0)
    adv_acc = accuracy_score(y_true, y_pred_adv)
    adv_p, adv_r, adv_f1, _ = precision_recall_fscore_support(y_true, y_pred_adv, average='weighted', zero_division=0)

    print("\n==================================================")
    print("         GENETIC ALGORITHM (GA) METRICS REPORT    ")
    print("==================================================")
    print(f"--- BASELINE (Before Attack) ---")
    print(f"Accuracy: {orig_acc * 100:.2f}% | Precision: {orig_p:.4f} | Recall: {orig_r:.4f} | F1: {orig_f1:.4f}")
    print(f"--- ADVERSARIAL (After Attack) ---")
    print(f"Accuracy: {adv_acc * 100:.2f}% | Precision: {adv_p:.4f} | Recall: {adv_r:.4f} | F1: {adv_f1:.4f}")
    print("==================================================\n")