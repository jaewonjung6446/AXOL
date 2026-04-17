"""Real-world text generator on a small English proverb corpus.

Measures four things on the same model:

  1. Held-out next-character accuracy  — generalisation to unseen text
  2. Held-out per-character perplexity — calibrated uncertainty
  3. Memorisation of training prompts  — exact recall
  4. Free generation from novel prompts — qualitative quality

Also runs a side-by-side comparison between:
  * LanguageModel            (single core, dim=28)
  * HierarchicalLanguageModel (4 levels, chunked pyramid)

Deliberately uses a public-domain corpus of short proverbs so the whole
experiment runs on CPU in under a minute.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from axol.quantum.hierarchical import HierarchicalLanguageModel, LevelConfig
from axol.quantum.language_model import LanguageModel


# ---------------------------------------------------------------------------
# Corpus — 50 public-domain English proverbs
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "a penny saved is a penny earned",
    "actions speak louder than words",
    "all that glitters is not gold",
    "better late than never",
    "birds of a feather flock together",
    "dont count your chickens before they hatch",
    "dont judge a book by its cover",
    "easy come easy go",
    "every cloud has a silver lining",
    "fortune favors the bold",
    "good things come to those who wait",
    "honesty is the best policy",
    "hope for the best prepare for the worst",
    "if it aint broke dont fix it",
    "in the land of the blind the one eyed man is king",
    "keep your friends close and your enemies closer",
    "laughter is the best medicine",
    "let sleeping dogs lie",
    "look before you leap",
    "many hands make light work",
    "necessity is the mother of invention",
    "no news is good news",
    "no pain no gain",
    "once bitten twice shy",
    "one mans trash is another mans treasure",
    "out of sight out of mind",
    "practice makes perfect",
    "rome was not built in a day",
    "slow and steady wins the race",
    "strike while the iron is hot",
    "the early bird catches the worm",
    "the grass is always greener on the other side",
    "the pen is mightier than the sword",
    "the proof of the pudding is in the eating",
    "there is no place like home",
    "there is no such thing as a free lunch",
    "time flies when youre having fun",
    "time heals all wounds",
    "too many cooks spoil the broth",
    "two heads are better than one",
    "two wrongs dont make a right",
    "when in rome do as the romans do",
    "when the going gets tough the tough get going",
    "where there is a will there is a way",
    "you cant have your cake and eat it too",
    "you cant teach an old dog new tricks",
    "you reap what you sow",
    "a bird in the hand is worth two in the bush",
    "a friend in need is a friend indeed",
    "a stitch in time saves nine",
]


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _tokenizer(model):
    return model.levels[0].verb, model.tokenizer if _is_hier(model) else (
        model.chat.verbalizer, model.chat.tokenizer)


def _get_tokenizer(model):
    return model.tokenizer if _is_hier(model) else model.chat.tokenizer


def next_char_accuracy(model, texts: list[str]) -> float:
    """Fraction of held-out characters where the greedy next-char matches."""
    tok = _get_tokenizer(model)
    correct = 0
    total = 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) < 2:
            continue
        _reset_wms(model)
        _ingest(model, ids[0])
        for j in range(1, len(ids)):
            pred = _predict_next_id(model)
            if pred == ids[j]:
                correct += 1
            total += 1
            _ingest(model, ids[j])
    return correct / max(total, 1)


def perplexity(model, texts: list[str]) -> float:
    """Per-character perplexity using the Verbalizer softmax as the LM head."""
    tok = _get_tokenizer(model)
    total_loss = 0.0
    total = 0
    for text in texts:
        ids = tok.encode(text)
        if len(ids) < 2:
            continue
        _reset_wms(model)
        _ingest(model, ids[0])
        for j in range(1, len(ids)):
            probs = _predict_next_probs(model)
            p = float(probs[ids[j]])
            p = max(p, 1e-9)
            total_loss += -math.log(p)
            total += 1
            _ingest(model, ids[j])
    if total == 0:
        return float("inf")
    return math.exp(total_loss / total)


# --- per-model adapters -----------------------------------------------------

def _is_hier(model) -> bool:
    return isinstance(model, HierarchicalLanguageModel)


def _reset_wms(model) -> None:
    if _is_hier(model):
        for L in model.levels:
            L.wm.reset()
            L.tokens_since_chunk = 0
    else:
        model.chat.working_memory.reset()


def _ingest(model, tid: int) -> None:
    """No-learn ingestion of one token id."""
    if _is_hier(model):
        model.ingest_token(int(tid), learn=False)
    else:
        emb = model.chat.verbalizer.encode_id(int(tid))
        model.chat.working_memory.add(emb)


def _predict_next_id(model) -> int:
    if _is_hier(model):
        L0 = model.levels[0]
        ctx = L0.wm.context() + model._conditioning_for_level(0)
        vec = L0.core.predict(ctx)
        tid, _ = L0.verb.decode_vec(vec)
    else:
        ctx = model.chat.working_memory.context()
        vec = model.chat.intuition.predict(ctx)
        tid, _ = model.chat.verbalizer.decode_vec(vec)
    return int(tid)


def _predict_next_probs(model) -> np.ndarray:
    """Softmax distribution over the vocab at the current state."""
    if _is_hier(model):
        L0 = model.levels[0]
        ctx = L0.wm.context() + model._conditioning_for_level(0)
        vec = L0.core.predict(ctx)
        return L0.verb.decode_distribution(vec, temperature=1.0)
    else:
        ctx = model.chat.working_memory.context()
        vec = model.chat.intuition.predict(ctx)
        return model.chat.verbalizer.decode_distribution(vec, temperature=1.0)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

EPOCHS = 5


def train_single(corpus: list[str], vocab: str) -> LanguageModel:
    lm = LanguageModel(
        vocab=vocab, embed_dim=20, window=14,
        regularization=1e-4, seed=0,
    )
    # Self-supervised char-level training, multiple passes to deepen moments.
    joined = " ".join(corpus)
    for _ in range(EPOCHS):
        lm.train_on_text(joined)
    return lm


def train_hier(corpus: list[str], vocab: str) -> HierarchicalLanguageModel:
    m = HierarchicalLanguageModel(
        vocab=vocab,
        levels=[
            LevelConfig(dim=20, window=12, chunk_size=6),  # surface — same dim as single
            LevelConfig(dim=14, window=8,  chunk_size=4),  # mid
            LevelConfig(dim=10, window=6,  chunk_size=3),  # upper
            LevelConfig(dim=6,  window=4,  chunk_size=0),  # top
        ],
        regularization=1e-4, seed=0,
    )
    joined = " ".join(corpus)
    for _ in range(EPOCHS):
        m.teach_text(joined, reset_wm=True)
    return m


def generate_single(model: LanguageModel, prompt: str, max_len: int = 50) -> str:
    res = model.generate(prompt, max_len=max_len, temperature=0.0)
    return res.text


def generate_hier(model: HierarchicalLanguageModel, prompt: str, max_len: int = 50) -> str:
    res = model.generate(prompt, max_len=max_len, temperature=0.0)
    return res.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(CORPUS))
    split = int(len(CORPUS) * 0.8)
    train_corpus = [CORPUS[i] for i in idx[:split]]
    test_corpus = [CORPUS[i] for i in idx[split:]]

    vocab_chars = sorted(set(" ".join(CORPUS)))
    vocab = "".join(vocab_chars)

    print(f"corpus  : {len(CORPUS)} proverbs, {sum(len(s) for s in CORPUS)} chars")
    print(f"vocab   : {len(vocab)} unique chars -> {vocab!r}")
    print(f"split   : train={len(train_corpus)}  test={len(test_corpus)}")
    print()

    # ---------- Single ----------
    print("=" * 75)
    print("Single-core LanguageModel (embed_dim=28, window=14)")
    print("=" * 75)
    t0 = time.perf_counter()
    lm_single = train_single(train_corpus, vocab)
    t_train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc_s = next_char_accuracy(lm_single, test_corpus)
    ppl_s = perplexity(lm_single, test_corpus)
    t_eval_s = time.perf_counter() - t0

    single_elems = 3 * lm_single.chat.intuition.ld ** 2
    print(f"  memory (G+H+B)          : {single_elems * 8 / 1024 / 1024:.2f} MB")
    print(f"  train time              : {t_train_s:.2f}s  "
          f"({lm_single.samples_trained} samples)")
    print(f"  held-out next-char acc  : {acc_s:.1%}    "
          f"(random baseline = {1/len(vocab):.1%})")
    print(f"  held-out perplexity     : {ppl_s:.2f}    "
          f"(uniform baseline = {len(vocab):.2f})")
    print(f"  eval time               : {t_eval_s:.2f}s")
    print()

    # ---------- Hierarchical ----------
    print("=" * 75)
    print("HierarchicalLanguageModel 4-level (surface=20, 14, 10, 6)")
    print("=" * 75)
    t0 = time.perf_counter()
    lm_hier = train_hier(train_corpus, vocab)
    t_train_h = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc_h = next_char_accuracy(lm_hier, test_corpus)
    ppl_h = perplexity(lm_hier, test_corpus)
    t_eval_h = time.perf_counter() - t0

    r = lm_hier.report()
    print(f"  memory (G+H+B)          : "
          f"{lm_hier.total_matrix_elements * 8 / 1024 / 1024:.2f} MB")
    print(f"  train time              : {t_train_h:.2f}s")
    print(f"  held-out next-char acc  : {acc_h:.1%}")
    print(f"  held-out perplexity     : {ppl_h:.2f}")
    print(f"  eval time               : {t_eval_h:.2f}s")
    print(f"  per-level dims          : {r.dims}")
    print(f"  per-level n_samples     : {r.n_samples}")
    print(f"  per-level Ω             : "
          f"{[f'{o:.3f}' for o in r.omegas]}")
    print()

    # ---------- Training-set memorisation ----------
    print("=" * 75)
    print("Training-set memorisation (single vs hier)")
    print("=" * 75)
    hits_s = hits_h = 0
    for proverb in train_corpus[:8]:
        words = proverb.split()
        prompt = " ".join(words[:2])
        tail = proverb[len(prompt):]
        gen_s = generate_single(lm_single, prompt + " ", max_len=len(tail) + 2)
        gen_h = generate_hier(lm_hier, prompt + " ", max_len=len(tail) + 2)
        ok_s = tail.strip().startswith(gen_s.strip()[:len(tail.strip())])
        ok_h = tail.strip().startswith(gen_h.strip()[:len(tail.strip())])
        hits_s += int(ok_s)
        hits_h += int(ok_h)
        print(f"  prompt {prompt!r:30} -> truth: {tail.strip()!r}")
        print(f"     single : {gen_s.strip()!r:45} {'✓' if ok_s else '✗'}")
        print(f"     hier   : {gen_h.strip()!r:45} {'✓' if ok_h else '✗'}")
    print(f"\n  train memorisation : single {hits_s}/8   hier {hits_h}/8")
    print()

    # ---------- Free generation from novel prompts ----------
    print("=" * 75)
    print("Free generation (novel prompts)")
    print("=" * 75)
    novel_prompts = [
        "the early bird ",
        "a stitch in ",
        "when in rome ",
        "dont count ",
        "slow and steady ",
    ]
    for p in novel_prompts:
        gen_s = generate_single(lm_single, p, max_len=40)
        gen_h = generate_hier(lm_hier, p, max_len=40)
        print(f"  prompt {p!r}")
        print(f"     single : {gen_s.strip()!r}")
        print(f"     hier   : {gen_h.strip()!r}")
    print()

    # ---------- Q&A-style training (AXOL's actual strong point) ----------
    print("=" * 75)
    print("Q&A-style training: prompt='first 2 words' -> 'remainder'")
    print("=" * 75)
    pairs: list[tuple[str, str]] = []
    for p in train_corpus:
        words = p.split()
        if len(words) < 3:
            continue
        pairs.append((" ".join(words[:2]), " " + " ".join(words[2:])))

    # Fresh models, trained only on the Q&A pairs
    lm_s2 = LanguageModel(
        vocab=vocab, embed_dim=20, window=14,
        regularization=1e-4, seed=0,
    )
    lm_s2.train_pairs(pairs, epochs=40)

    lm_h2 = HierarchicalLanguageModel(
        vocab=vocab,
        levels=[
            LevelConfig(dim=20, window=12, chunk_size=6),
            LevelConfig(dim=14, window=8,  chunk_size=4),
            LevelConfig(dim=10, window=6,  chunk_size=3),
            LevelConfig(dim=6,  window=4,  chunk_size=0),
        ],
        regularization=1e-4, seed=0,
    )
    lm_h2.teach_pairs(pairs, epochs=40)

    hits_s2 = hits_h2 = 0
    for q, expected in pairs[:10]:
        gen_s = lm_s2.generate(q, max_len=len(expected) + 3,
                               temperature=0.0).text
        gen_h = lm_h2.generate(q, max_len=len(expected) + 3,
                               temperature=0.0).text
        ok_s = gen_s.strip().startswith(expected.strip())
        ok_h = gen_h.strip().startswith(expected.strip())
        hits_s2 += int(ok_s)
        hits_h2 += int(ok_h)
        print(f"  {q!r:20} -> {expected.strip()!r}")
        print(f"      single: {gen_s.strip()!r:45} {'✓' if ok_s else '✗'}")
        print(f"      hier  : {gen_h.strip()!r:45} {'✓' if ok_h else '✗'}")
    print(f"\n  Q&A memorisation : single {hits_s2}/10    hier {hits_h2}/10")
    print()

    # ---------- Summary ----------
    print("=" * 75)
    print("Summary")
    print("=" * 75)
    print(f"{'':28s}  {'Single':>12s}  {'Hierarchical':>14s}")
    print(f"{'memory (MB)':28s}  {single_elems*8/1024/1024:>12.2f}  "
          f"{lm_hier.total_matrix_elements*8/1024/1024:>14.2f}")
    print(f"{'train time (s)':28s}  {t_train_s:>12.2f}  {t_train_h:>14.2f}")
    print(f"{'next-char accuracy':28s}  {acc_s:>12.1%}  {acc_h:>14.1%}")
    print(f"{'perplexity':28s}  {ppl_s:>12.2f}  {ppl_h:>14.2f}")
    print(f"{'train memorisation':28s}  {hits_s:>12d}  {hits_h:>14d}")
    print(f"{'Q&A memorisation (/10)':28s}  {hits_s2:>12d}  {hits_h2:>14d}")


if __name__ == "__main__":
    main()
