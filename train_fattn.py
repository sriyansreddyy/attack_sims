"""
train_fattn.py
==============
Trains the pre-attack model Fattn on IMDB (public dataset).

Paper Section V-A-4:
  - Trained EXCLUSIVELY on IMDB (distinct from all target model training data)
  - LR = 1e-3, epochs = 100
  - GloVe glove.6B.200d + static BERT embeddings

Run AFTER bert_imdb_finetune.py (needs saved_model/ for BERT tokenizer).

    python train_fattn.py --glove path/to/glove.6B.200d.txt

Output:
    ./fattn_checkpoint.pt   <- weights loaded by run_benchmarks.py
"""

import argparse, os, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from f2attack_core import Fattn, EmbeddingProvider

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--glove",     required=True,  help="Path to glove.6B.200d.txt")
parser.add_argument("--bert_dir",  default="./bert-base-uncased-imdb")
parser.add_argument("--epochs",    type=int, default=100)
parser.add_argument("--lr",        type=float, default=1e-3)
parser.add_argument("--batch",     type=int, default=32)
parser.add_argument("--max_len",   type=int, default=200)
parser.add_argument("--out",       default="./fattn_checkpoint.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Embeddings ────────────────────────────────────────────────────────────────
emb = EmbeddingProvider(args.glove, args.bert_dir, device=str(device))

# ── Dataset ───────────────────────────────────────────────────────────────────
class IMDBEmbeddingDataset(Dataset):
    def __init__(self, split, max_len):
        raw        = load_dataset("imdb")[split]
        self.texts  = raw["text"]
        self.labels = raw["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words  = self.texts[idx].split()[: self.max_len]
        g, b   = emb.encode_text(words)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return g, b, label

def collate_fn(batch):
    gs, bs, labels = zip(*batch)
    max_t = max(g.shape[0] for g in gs)
    g_pad = torch.zeros(len(gs), max_t, gs[0].shape[-1])
    b_pad = torch.zeros(len(bs), max_t, bs[0].shape[-1])
    for i, (g, b) in enumerate(zip(gs, bs)):
        g_pad[i, :g.shape[0]] = g
        b_pad[i, :b.shape[0]] = b
    return g_pad, b_pad, torch.stack(labels)

train_ds = IMDBEmbeddingDataset("train", args.max_len)
val_ds   = IMDBEmbeddingDataset("test",  args.max_len)

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Fattn(glove_dim=200, bert_dim=768, hidden_dim=150, num_classes=2).to(device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)
crit  = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    # Train
    model.train()
    total_loss = 0.0
    for g, b, y in train_dl:
        g, b, y = g.to(device), b.to(device), y.to(device)
        optim.zero_grad()
        logits = model(g, b)
        loss   = crit(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()

    # Validate every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for g, b, y in val_dl:
                g, b, y = g.to(device), b.to(device), y.to(device)
                preds   = model(g, b).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        acc = correct / total * 100
        print(f"Epoch {epoch:3d} | Loss {total_loss/len(train_dl):.4f} | Val Acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ Saved best model (acc={best_acc:.2f}%)")

print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")
print(f"Fattn saved to: {args.out}")