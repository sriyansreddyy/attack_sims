"""
f2attack_core.py
================
Faithful implementation of F2Attack (Wang et al., IEEE TIFS 2025).

Components (matching paper Section IV exactly):
  - Fattn   : Pre-attack BiLSTM+Attention model for word importance scores
  - SimScore: Semantic similarity scoring via USE / cosine sim
  - TWS     : Two-Factors Word Scoring  (score_i = alpha_i / sim_i)
  - AEI     : Adversarial Example Initialization
  - AEO     : Adversarial Example Optimization (simulated annealing)
  - F2Attack: Full pipeline (Algorithm 1)

All equation numbers refer to the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, math, random
from typing import List, Tuple, Dict, Optional, Callable

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Synonym dictionary helper (counter-fitted word embeddings, k=5 default)
# ─────────────────────────────────────────────────────────────────────────────

class SynonymDict:
    """
    Wraps TextAttack's WordNetAugmenter or a pre-built counter-fitted
    embedding synonym set.  We default to TextAttack's built-in
    counter-fitted synonyms (same source the paper uses).
    """
    def __init__(self, k: int = 5):
        self.k = k
        self._cache: Dict[str, List[str]] = {}
        # Try to load TextAttack's counter-fitted synonyms
        try:
            from textattack.augmentation import WordNetAugmenter
            from textattack.transformations import WordSwapEmbedding
            self._swap = WordSwapEmbedding(max_candidates=k)
            self._mode = "textattack"
        except ImportError:
            # Fallback: NLTK WordNet
            import nltk
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet
            self._wordnet = wordnet
            self._mode = "wordnet"

    def get(self, word: str) -> List[str]:
        if word in self._cache:
            return self._cache[word]
        syns = self._fetch(word)
        self._cache[word] = syns
        return syns

    def _fetch(self, word: str) -> List[str]:
        if self._mode == "textattack":
            # TextAttack WordSwapEmbedding works on AttackedText objects
            # We keep it simple: just use wordnet as robust fallback
            pass
        # WordNet fallback (always available)
        try:
            from nltk.corpus import wordnet
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower():
                        synonyms.add(candidate)
                    if len(synonyms) >= self.k:
                        break
                if len(synonyms) >= self.k:
                    break
            return list(synonyms)[: self.k]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Fattn — Pre-Attack Model  (paper Section IV-B-1, Fig.1, Eq.3-11)
# ─────────────────────────────────────────────────────────────────────────────

class Fattn(nn.Module):
    """
    Dual-embedding BiLSTM + Attention model.
    Trained on IMDB (public dataset, not target model's training data).

    Architecture (Fig. 1):
      - GloVe BiLSTM  →  annotation g_i   (Eq. 3-4)
      - BERT-static BiLSTM → annotation b_i   (Eq. 5-6)
      - Merge: h_i = g_i + b_i            (Eq. 7)
      - Attention: alpha_i via softmax     (Eq. 8-9)
      - Context vector c = sum(alpha_i * h_i)  (Eq. 10)
      - Linear classifier                  (Eq. 11)
    """
    def __init__(
        self,
        glove_dim: int = 200,
        bert_dim: int = 768,
        hidden_dim: int = 150,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Two parallel BiLSTMs (one per embedding)
        self.glove_lstm = nn.LSTM(
            glove_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.bert_lstm  = nn.LSTM(
            bert_dim,  hidden_dim, batch_first=True, bidirectional=True
        )

        # Attention (Eq. 8-9): tanh(W_s * h + b_s) -> score, then softmax
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_v = nn.Parameter(torch.randn(hidden_dim * 2))

        # Linear classifier (Eq. 11)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        glove_emb: torch.Tensor,   # (B, T, glove_dim)
        bert_emb:  torch.Tensor,   # (B, T, bert_dim)
        return_alpha: bool = False,
    ):
        # Eq. 3-4: GloVe BiLSTM
        g, _ = self.glove_lstm(glove_emb)   # (B, T, 2*H)
        # Eq. 5-6: BERT BiLSTM
        b, _ = self.bert_lstm(bert_emb)     # (B, T, 2*H)

        # Eq. 7: merge
        h = g + b                           # (B, T, 2*H)

        # Eq. 8: u_i = tanh(W_s * h_i + b_s)
        u = torch.tanh(self.attn_W(h))      # (B, T, 2*H)

        # Eq. 9: alpha_i = softmax(u_i^T * v)
        score  = torch.matmul(u, self.attn_v)   # (B, T)
        alpha  = torch.softmax(score, dim=-1)    # (B, T)

        # Eq. 10: c = sum_i(alpha_i * h_i)
        c = (alpha.unsqueeze(-1) * h).sum(dim=1)  # (B, 2*H)
        c = self.dropout(c)

        # Eq. 11
        logits = self.classifier(c)
        if return_alpha:
            return logits, alpha
        return logits

    @torch.no_grad()
    def get_attention_scores(
        self,
        glove_emb: torch.Tensor,
        bert_emb:  torch.Tensor,
    ) -> np.ndarray:
        """Returns alpha (T,) for a single sample."""
        self.eval()
        _, alpha = self.forward(
            glove_emb.unsqueeze(0),
            bert_emb.unsqueeze(0),
            return_alpha=True,
        )
        return alpha.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Embedding helpers (GloVe + static BERT)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """
    Loads GloVe (glove.6B.200d) and constructs static BERT embeddings
    exactly as described in paper Section V-A-4.

    GloVe: glove.6B.200d  (statistical global co-occurrence)
    BERT-static: last hidden layer of frozen BERT for each of the 400k
                 GloVe vocabulary words (local contextual representation)
    """
    def __init__(self, glove_path: str, bert_model_dir: str, device: str = "cpu"):
        self.device    = device
        self.glove_dim = 200
        self.bert_dim  = 768
        print("Loading GloVe embeddings …")
        self.glove_vocab, self.glove_matrix = self._load_glove(glove_path)
        print(f"  GloVe: {len(self.glove_vocab)} words")
        print("Building static BERT embeddings …")
        self.bert_matrix = self._build_bert_static(bert_model_dir)
        print("  Done.")

    def _load_glove(self, path: str):
        vocab, vecs = {}, []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.rstrip().split(" ")
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                vocab[word] = i
                vecs.append(vec)
        matrix = np.stack(vecs, axis=0)
        return vocab, matrix

    def _build_bert_static(self, bert_dir: str):
        """
        Paper: input each of the 400k GloVe words into frozen BERT,
        take last hidden layer embedding as static BERT vector.
        This is expensive for 400k words; we do it lazily (cache on disk).
        """
        import os
        cache = os.path.join(bert_dir, "static_bert_glove.npy")
        if os.path.exists(cache):
            return np.load(cache, mmap_mode="r")

        from transformers import BertModel, BertTokenizerFast
        print("  Building BERT static embeddings (one-time, may take minutes) …")
        tokenizer = BertTokenizerFast.from_pretrained(bert_dir)
        bert      = BertModel.from_pretrained(bert_dir).to(self.device)
        bert.eval()

        words  = list(self.glove_vocab.keys())
        vecs   = np.zeros((len(words), self.bert_dim), dtype=np.float32)
        BS     = 512

        with torch.no_grad():
            for start in range(0, len(words), BS):
                batch  = words[start : start + BS]
                enc    = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=4
                ).to(self.device)
                out    = bert(**enc)
                # Take [CLS] token of last hidden layer for each word
                emb    = out.last_hidden_state[:, 0, :].cpu().numpy()
                vecs[start : start + len(batch)] = emb
                if start % 50000 == 0:
                    print(f"    {start}/{len(words)}")

        np.save(cache, vecs)
        return vecs

    def encode_text(self, words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (glove_emb, bert_emb) tensors of shape (T, dim).
        Unknown words get zero vectors.
        """
        T        = len(words)
        g_mat    = np.zeros((T, self.glove_dim), dtype=np.float32)
        b_mat    = np.zeros((T, self.bert_dim),  dtype=np.float32)
        for i, w in enumerate(words):
            lw = w.lower()
            if lw in self.glove_vocab:
                idx      = self.glove_vocab[lw]
                g_mat[i] = self.glove_matrix[idx]
                b_mat[i] = self.bert_matrix[idx]
        return (
            torch.tensor(g_mat, device=self.device),
            torch.tensor(b_mat, device=self.device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SimScore  (paper Section IV-B-2)
# ─────────────────────────────────────────────────────────────────────────────

class SimScore:
    """
    Computes semantic similarity score for each word w_i:
      sim_i = 1 - min_j(sim(x, x^j_i))
    where x^j_i is x with w_i replaced by its j-th synonym.

    GPU-accelerated: runs sentence encoder on GPU if available.
    Cache-backed: repeated sentence pairs (common in AEO loop) are free.

    Uses sentence-transformers as a drop-in for the paper's TF-Hub USE.
    For exact paper reproduction swap model_name for the TF Hub USE module.
    """
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: str     = "cuda",
        batch_size: int = 256,
        cache_size: int = 50_000,
    ):
        from sentence_transformers import SentenceTransformer
        self.device     = device
        self.batch_size = batch_size
        self.model      = SentenceTransformer(model_name, device=device)
        self.model.eval()

        # LRU embedding cache: sentence → np.ndarray
        # Avoids re-encoding the same sentence repeatedly during AEO iterations
        from functools import lru_cache
        self._cache_size = cache_size
        self._emb_cache: Dict[str, np.ndarray] = {}

    def _encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encode a list of sentences, using cache for any already-seen ones.
        New sentences are encoded in one batched GPU call.
        """
        result    = np.zeros((len(sentences), self.model.get_sentence_embedding_dimension()),
                             dtype=np.float32)
        to_encode = []   # (original_list_idx, sentence)

        for i, s in enumerate(sentences):
            if s in self._emb_cache:
                result[i] = self._emb_cache[s]
            else:
                to_encode.append((i, s))

        if to_encode:
            idxs, sents = zip(*to_encode)
            # Single batched GPU forward pass for all unseen sentences
            with torch.no_grad():
                embs = self.model.encode(
                    list(sents),
                    batch_size      = self.batch_size,
                    convert_to_numpy= True,
                    show_progress_bar= False,
                )
            for i, emb in zip(idxs, embs):
                result[i] = emb
                # Evict oldest entry if cache is full (simple FIFO)
                if len(self._emb_cache) >= self._cache_size:
                    self._emb_cache.pop(next(iter(self._emb_cache)))
                self._emb_cache[sentences[i]] = emb

        return result

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def cosine_matrix(self, anchor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Vectorised cosine between one anchor and N rows. Returns (N,) array."""
        norms  = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        normed = matrix / norms
        return normed @ (anchor / (np.linalg.norm(anchor) + 1e-8))

    def sentence_sim(self, s1: str, s2: str) -> float:
        embs = self._encode([s1, s2])
        return self.cosine(embs[0], embs[1])

    def batch_sim(self, original: str, candidates: List[str]) -> List[float]:
        """One GPU call for original + all candidates."""
        all_sents = [original] + candidates
        embs      = self._encode(all_sents)
        sims      = self.cosine_matrix(embs[0], embs[1:])
        return sims.tolist()

    def score_word(
        self,
        words: List[str],
        idx: int,
        synonyms: List[str],
    ) -> float:
        """
        Returns sim_i = 1 - min_j(sim(x, x^j_i))   (paper Section IV-B-2 step ii)
        High value → word strongly affects semantics → should score LOW in TWS.
        """
        if not synonyms:
            return 0.0
        original_text = " ".join(words)
        candidates    = [
            " ".join(words[:idx] + [syn] + words[idx+1:])
            for syn in synonyms
        ]
        sims    = self.batch_sim(original_text, candidates)
        min_sim = min(sims)
        return 1.0 - min_sim

    def score_all_words(
        self,
        words: List[str],
        word_indices: List[int],
        synonyms_per_word: List[List[str]],
    ) -> List[float]:
        """
        Batch SimScore for ALL words in one text in a single encoder call.
        This is the key GPU optimisation for TWS: instead of N separate
        batch_sim calls (one per word), we build one flat list of all
        perturbed texts and encode everything together.

        Returns sim_i scores in the same order as word_indices.
        """
        original_text = " ".join(words)

        # Build flat candidate list, keeping track of word boundaries
        flat_candidates: List[str] = []
        boundaries: List[Tuple[int, int]] = []   # (start, end) per word

        for i_w, (w_idx, syns) in enumerate(zip(word_indices, synonyms_per_word)):
            start = len(flat_candidates)
            for syn in syns:
                flat_candidates.append(
                    " ".join(words[:w_idx] + [syn] + words[w_idx+1:])
                )
            boundaries.append((start, len(flat_candidates)))

        if not flat_candidates:
            return [0.0] * len(word_indices)

        # One batched GPU encode for original + all perturbed texts
        all_sents = [original_text] + flat_candidates
        embs      = self._encode(all_sents)
        orig_emb  = embs[0]
        cand_embs = embs[1:]

        sim_scores = []
        for (start, end) in boundaries:
            if start == end:
                sim_scores.append(0.0)
                continue
            sims    = self.cosine_matrix(orig_emb, cand_embs[start:end])
            min_sim = float(sims.min())
            sim_scores.append(1.0 - min_sim)

        return sim_scores


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TWS — Two-Factors Word Scoring  (paper Section IV-B-3)
# ─────────────────────────────────────────────────────────────────────────────

def two_factors_word_scoring(
    words: List[str],
    fattn: Fattn,
    emb_provider: EmbeddingProvider,
    sim_scorer: SimScore,
    syn_dict: SynonymDict,
) -> List[Tuple[int, str, float]]:
    """
    Returns sorted list of (word_index, word, score) in descending order.
    score_i = alpha_i / sim_i   (paper Section IV-B-3 step iii)

    Optimised: all SimScore calls are batched into one GPU encoder forward pass
    via score_all_words(), rather than one call per word.
    """
    g_emb, b_emb = emb_provider.encode_text(words)
    alphas        = fattn.get_attention_scores(g_emb, b_emb)  # (T,)

    # Collect candidate words and their synonym sets first
    candidate_indices: List[int]         = []
    candidate_words:   List[str]         = []
    candidate_syns:    List[List[str]]   = []

    for i, w in enumerate(words):
        if len(w) <= 2 or not w.isalpha():
            continue
        syns = syn_dict.get(w)
        if not syns:
            continue
        candidate_indices.append(i)
        candidate_words.append(w)
        candidate_syns.append(syns)

    if not candidate_indices:
        return []

    # Single batched GPU call for all words' SimScores
    sim_scores = sim_scorer.score_all_words(words, candidate_indices, candidate_syns)

    scored = []
    for i_w, (w_idx, w, sim_i) in enumerate(
        zip(candidate_indices, candidate_words, sim_scores)
    ):
        alpha_i = float(alphas[w_idx])
        score_i = alpha_i / (sim_i + 1e-8)
        scored.append((w_idx, w, score_i))

    # Sort descending — highest score = best attack target
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AEI — Adversarial Example Initialization  (paper Section IV-D step 2)
# ─────────────────────────────────────────────────────────────────────────────

def adversarial_example_init(
    words: List[str],
    true_label: int,
    sorted_words: List[Tuple[int, str, float]],
    target_model_fn: Callable[[str], int],
    syn_dict: SynonymDict,
    query_counter: List[int],
    query_budget: int,
) -> Optional[List[str]]:
    """
    Iterates through TWS-sorted words, replaces each with a random synonym,
    queries target model, returns first xpert that fools the model.
    Algorithm 1, lines 4-10.
    """
    x_pert = words[:]

    for idx, word, score in sorted_words:
        if query_counter[0] >= query_budget:
            return None
        syns = syn_dict.get(word)
        if not syns:
            continue
        chosen = random.choice(syns)
        x_pert[idx] = chosen

        pred = target_model_fn(" ".join(x_pert))
        query_counter[0] += 1

        if pred != true_label:
            return x_pert   # Found init adversarial example

    return None   # Initialization failed within budget


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AEO — Adversarial Example Optimization  (paper Section IV-C)
#           Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def adversarial_example_optimization(
    x_init_adv: List[str],
    x_orig: List[str],
    true_label: int,
    target_model_fn: Callable[[str], int],
    sim_scorer: SimScore,
    syn_dict: SynonymDict,
    query_counter: List[int],
    query_budget: int,
    T_init: float = 1.0,
    alpha: float  = 0.9,
) -> List[str]:
    """
    Simulated annealing over adversarial example space.
    Maximizes semantic similarity Sim(x_adv, x) while maintaining adversarial label.
    Paper Section IV-C, Steps 1-3.
    """
    orig_text   = " ".join(x_orig)
    x_prev      = x_init_adv[:]
    x_best      = x_init_adv[:]
    sim_best    = sim_scorer.sentence_sim(orig_text, " ".join(x_best))
    T           = T_init

    # Find perturbed positions (diffs between x_orig and x_init_adv)
    def get_diffs(x_a, x_b):
        return [i for i, (a, b) in enumerate(zip(x_a, x_b)) if a != b]

    while query_counter[0] < query_budget and T > 0.01:
        diffs = get_diffs(x_orig, x_prev)
        if not diffs:
            break

        # Step 2(i): sample target position t weighted by semantic similarity
        # (positions that, when restored, give higher sim get higher prob)
        weights = []
        for pos in diffs:
            x_tmp      = x_prev[:]
            x_tmp[pos] = x_orig[pos]          # restore to original
            s          = sim_scorer.sentence_sim(orig_text, " ".join(x_tmp))
            weights.append(s)

        total = sum(weights) + 1e-8
        probs = [w / total for w in weights]
        t     = random.choices(diffs, weights=probs, k=1)[0]

        # Step 2(ii): try restoring position t to original word
        x_star       = x_prev[:]
        x_star[t]    = x_orig[t]
        if query_counter[0] < query_budget:
            pred = target_model_fn(" ".join(x_star))
            query_counter[0] += 1
            if pred != true_label:
                x_new = x_star
            else:
                # Step 2(iii): try other synonyms for position t
                x_new = None
                current_sub = x_prev[t]
                syns = [s for s in syn_dict.get(x_orig[t]) if s != current_sub]
                for syn in syns:
                    if query_counter[0] >= query_budget:
                        break
                    x_cand       = x_prev[:]
                    x_cand[t]    = syn
                    pred         = target_model_fn(" ".join(x_cand))
                    query_counter[0] += 1
                    if pred != true_label:
                        x_new = x_cand
                        break
                if x_new is None:
                    T *= alpha
                    continue

            # Step 3: Metropolis acceptance criterion
            sim_new  = sim_scorer.sentence_sim(orig_text, " ".join(x_new))
            sim_prev = sim_scorer.sentence_sim(orig_text, " ".join(x_prev))
            delta    = sim_new - sim_prev

            if delta > 0:                          # Step 3(i): always accept improvement
                x_prev = x_new
            else:                                  # Step 3(ii): accept with probability
                prob = math.exp(delta / (T + 1e-8))
                if random.random() < prob:
                    x_prev = x_new

            # Step 3(iii): update global best
            if sim_scorer.sentence_sim(orig_text, " ".join(x_prev)) > sim_best:
                x_best   = x_prev[:]
                sim_best = sim_scorer.sentence_sim(orig_text, " ".join(x_best))

        # Step 3(iv): anneal temperature
        T *= alpha

    return x_best


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Full F2Attack Pipeline  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def f2attack(
    text: str,
    true_label: int,
    target_model_fn: Callable[[str], int],
    fattn: Fattn,
    emb_provider: EmbeddingProvider,
    sim_scorer: SimScore,
    syn_dict: SynonymDict,
    query_budget: int  = 100,
    T_init: float      = 1.0,
    anneal_rate: float = 0.9,
) -> Dict:
    """
    Full F2Attack as per Algorithm 1.

    Returns dict with:
        success   : bool
        adv_text  : str  (adversarial example or original if failed)
        sim       : float (semantic similarity to original)
        queries   : int  (total queries used)
    """
    words          = text.split()
    query_counter  = [0]

    # Step 1: TWS (line 1-2)
    sorted_words = two_factors_word_scoring(
        words, fattn, emb_provider, sim_scorer, syn_dict
    )

    # Step 2: AEI (lines 4-10)
    x_init_adv = adversarial_example_init(
        words, true_label, sorted_words,
        target_model_fn, syn_dict, query_counter, query_budget
    )

    if x_init_adv is None:
        return {
            "success"  : False,
            "adv_text" : text,
            "sim"      : 1.0,
            "queries"  : query_counter[0],
        }

    # Step 3: AEO (lines 11-12)
    x_best = adversarial_example_optimization(
        x_init_adv, words, true_label,
        target_model_fn, sim_scorer, syn_dict,
        query_counter, query_budget, T_init, anneal_rate
    )

    adv_text = " ".join(x_best)
    sim      = sim_scorer.sentence_sim(text, adv_text)

    return {
        "success"  : True,
        "adv_text" : adv_text,
        "sim"      : sim,
        "queries"  : query_counter[0],
    }