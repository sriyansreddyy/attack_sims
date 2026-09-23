import os
import torch
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import textattack

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import datasets.utils.info_utils
import datasets.builder
datasets.utils.info_utils.verify_splits = lambda *args, **kwargs: None
datasets.builder.verify_splits = lambda *args, **kwargs: None
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
local_model_path = os.path.abspath("bert-base-uncased-imdb")
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True).to(device)

model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

raw_dataset = load_dataset("imdb", split="test")
dataset_elements = [(sample["text"], sample["label"]) for sample in raw_dataset]
textattack_dataset = textattack.datasets.Dataset(dataset_elements)

# FIX: BAE Recipe Name Correction
attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)

attack_args = textattack.AttackArgs(
    num_examples=1000,
    log_to_csv="results_bae.csv",
    checkpoint_interval=100,
    checkpoint_dir="checkpoints_bae",
    disable_stdout=False
)

print("[START] BAE Attack running...")
attacker = textattack.Attacker(attack, textattack_dataset, attack_args)
results = attacker.attack_dataset()

# --- METRICS ENGINE ---
y_true, y_pred_orig, y_pred_adv = [], [], []
def extract_val(val):
    return val.item() if hasattr(val, 'item') else val

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
print("          BAE METRICS REPORT                      ")
print("==================================================")
print(f"--- BASELINE (Before Attack) ---")
print(f"Accuracy: {orig_acc * 100:.2f}% | Precision: {orig_p:.4f} | Recall: {orig_r:.4f} | F1: {orig_f1:.4f}")
print(f"--- ADVERSARIAL (After Attack) ---")
print(f"Accuracy: {adv_acc * 100:.2f}% | Precision: {adv_p:.4f} | Recall: {adv_r:.4f} | F1: {adv_f1:.4f}")
print("==================================================\n")