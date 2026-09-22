import os
import torch
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)

# ==============================================================================
# --- PHASE 2 & 3: GLOBAL FLAIR INTERCEPTION & LINGUISTIC TAXONOMY PATCH ---
# ==============================================================================
import flair.models

class NLTKSequenceTagger:
    """Mock class replacing Flair Neural Network with NLTK Universal POS Tagging.
    Includes memory-caching to eliminate redundant CPU-bound evaluations.
    """
    def __init__(self):
        self._cache = {}

    def predict(self, sentences, **kwargs):
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sentence in sentences:
            # Convert to hashable tuple for dictionary caching
            words = tuple(token.text for token in sentence)
            
            if words not in self._cache:
                self._cache[words] = nltk.pos_tag(words, tagset='universal')
            
            tags = self._cache[words]
            
            for token, (_, tag) in zip(sentence, tags):
                token.add_tag('pos', tag)
                token.add_tag('upos', tag)

# Bind the mock infrastructure to the global module namespace
flair.models.SequenceTagger.load = lambda *args, **kwargs: NLTKSequenceTagger()
# ==============================================================================

import textattack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Force local file validation and disable external huggingface metadata splits checks
import datasets.utils.info_utils
import datasets.builder
datasets.utils.info_utils.verify_splits = lambda *args, **kwargs: None
datasets.builder.verify_splits = lambda *args, **kwargs: None
from datasets import load_dataset

# Hardware Routing
device = "cuda" if torch.cuda.is_available() else "cpu"
local_model_path = os.path.abspath("bert-base-uncased-imdb")

# Model Instantiation
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True).to(device)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Dataset Pipeline Loading & Sequence Truncation Constraint
raw_dataset = load_dataset("imdb", split="test")

dataset_elements = []
# Filter out computationally prohibitive long-tail sequences (L > 150 words)
for sample in raw_dataset:
    if len(sample["text"].split()) <= 150:
        dataset_elements.append((sample["text"], sample["label"]))

textattack_dataset = textattack.datasets.Dataset(dataset_elements)

# Particle Swarm Optimization Recipe Initialization (Zang et al., 2020)
attack = textattack.attack_recipes.PSOZang2020.build(model_wrapper)

# Runtime Execution Arguments Setup - Bounded Constraints Applied
attack_args = textattack.AttackArgs(
    num_examples=1000,
    log_to_csv="results_pso.csv",
    checkpoint_interval=100,
    checkpoint_dir="checkpoints_pso",
    disable_stdout=False,
    query_budget=400  # Enforces a hard limit on forward passes per document
)

print("[START] Hardened PSO Adversarial Attack Sequence Initiated...")
attacker = textattack.Attacker(attack, textattack_dataset, attack_args)
results = attacker.attack_dataset()

# ==============================================================================
# --- POST-ATTACK METRICS EVALUATION ENGINE ---
# ==============================================================================
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

# Mathematical Assessment
orig_acc = accuracy_score(y_true, y_pred_orig)
orig_p, orig_r, orig_f1, _ = precision_recall_fscore_support(y_true, y_pred_orig, average='weighted', zero_division=0)
adv_acc = accuracy_score(y_true, y_pred_adv)
adv_p, adv_r, adv_f1, _ = precision_recall_fscore_support(y_true, y_pred_adv, average='weighted', zero_division=0)

print("\n" + "="*50)
print("             PSO METRICS REPORT                    ")
print("="*50)
print(f"--- BASELINE PERFORMANCE (Pre-Attack) ---")
print(f"Accuracy: {orig_acc * 100:.2f}% | Precision: {orig_p:.4f} | Recall: {orig_r:.4f} | F1-Score: {orig_f1:.4f}")
print(f"--- ADVERSARIAL PERFORMANCE (Post-Attack) ---")
print(f"Accuracy: {adv_acc * 100:.2f}% | Precision: {adv_p:.4f} | Recall: {adv_r:.4f} | F1-Score: {adv_f1:.4f}")
print("="*50 + "\n")