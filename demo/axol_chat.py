"""Interactive CLI for the AXOL language model.

Examples
--------
Fresh session, supervised Q&A, interactive:
    python -m demo.axol_chat

Train on a text file then chat:
    python -m demo.axol_chat --train corpus.txt --save mymodel

Load a saved model and chat:
    python -m demo.axol_chat --load mymodel

One-shot generation (no REPL):
    python -m demo.axol_chat --load mymodel --prompt "hello" --no-repl
"""

from __future__ import annotations

import argparse
import sys

# Ensure the repo root is importable when run as a script.
_THIS = __file__
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(_THIS)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from axol.quantum.language_model import LanguageModel  # noqa: E402


DEFAULT_PAIRS = [
    ("hi", "hello, how are you"),
    ("hello", "hi there"),
    ("how are you", "i am fine, thanks"),
    ("bye", "goodbye"),
    ("thanks", "you are welcome"),
    ("what is axol", "an ai with intuition and verbalisation"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="axol-chat",
        description="AXOL conversational language model — REPL + trainer",
    )
    p.add_argument("--load", metavar="PATH", help="load saved model from PATH")
    p.add_argument("--save", metavar="PATH", help="save trained model to PATH")
    p.add_argument("--train", metavar="FILE", help="train on text file (self-supervised)")
    p.add_argument(
        "--seed-qa",
        action="store_true",
        help="pre-train with a small built-in Q&A set (useful for fresh models)",
    )
    p.add_argument("--prompt", help="one-shot prompt; skips the REPL unless --repl")
    p.add_argument("--max-len", type=int, default=80, help="max response length")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy, >0 = softmax sampling")
    p.add_argument("--top-k", type=int, default=None,
                   help="top-k sampling (requires --temperature>0)")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="abort a generation step if cosine drops below this")
    p.add_argument("--embed-dim", type=int, default=24)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--epochs", type=int, default=30,
                   help="repetitions for --seed-qa")
    p.add_argument("--no-repl", action="store_true",
                   help="do not start interactive mode")
    p.add_argument("--repl", action="store_true",
                   help="force interactive mode even after --prompt")
    return p


def run() -> int:
    args = build_parser().parse_args()

    if args.load:
        lm = LanguageModel.load(args.load)
        print(f"[loaded] {args.load}  (samples trained = {lm.samples_trained})")
    else:
        lm = LanguageModel(embed_dim=args.embed_dim, window=args.window)
        print(f"[fresh] embed_dim={args.embed_dim} window={args.window}")

    if args.seed_qa:
        lm.train_pairs(DEFAULT_PAIRS, epochs=args.epochs)
        print(f"[seed-qa] trained {len(DEFAULT_PAIRS)} pairs x {args.epochs} epochs "
              f"→ samples={lm.samples_trained}")

    if args.train:
        with open(args.train, "r", encoding="utf-8") as f:
            corpus = f.read()
        n = lm.train_on_text(corpus)
        print(f"[train] {args.train}: absorbed {n} char-level samples")

    if args.save:
        lm.save(args.save)
        print(f"[saved] {args.save}.npz + {args.save}.json")

    def _generate(prompt: str) -> None:
        res = lm.generate(
            prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            min_confidence=args.min_confidence,
            seed=None,
        )
        avg_conf = (sum(res.confidences) / len(res.confidences)
                    if res.confidences else 0.0)
        print(f"AXOL: {res.text}")
        print(f"  [stopped={res.stopped_reason} "
              f"Ω={res.omega:.3f} Φ={res.phi:.3f} avg_conf={avg_conf:.3f}]")

    if args.prompt:
        _generate(args.prompt)

    # REPL decision: default to REPL unless --prompt without --repl, or --no-repl.
    go_repl = args.repl or (not args.prompt and not args.no_repl)
    if not go_repl:
        return 0

    print("\nAXOL chat — type '/quit' to exit, '/help' for commands.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        s = line.strip()
        if not s:
            continue
        if s in ("/quit", "/exit"):
            break
        if s == "/help":
            print("  /quit                  exit")
            print("  /teach IN -> OUT       add a supervised pair")
            print("  /forget FACTOR         decay all memory by FACTOR in [0,1]")
            print("  /save PATH             save current model")
            print("  /report                show Ω, Φ, samples")
            print("  (anything else)        prompt -> response")
            continue
        if s.startswith("/teach"):
            body = s[len("/teach"):].strip()
            if "->" in body:
                left, right = body.split("->", 1)
                lm.chat_once(left.strip(), right.strip(), learn=True)
                print(f"  [taught] {left.strip()!r} -> {right.strip()!r}")
            else:
                print("  usage: /teach INPUT -> OUTPUT")
            continue
        if s.startswith("/forget"):
            body = s[len("/forget"):].strip()
            try:
                factor = float(body)
            except ValueError:
                print("  usage: /forget FACTOR (0..1)")
                continue
            lm.forget(factor)
            print(f"  [forgot] factor={factor}")
            continue
        if s.startswith("/save"):
            body = s[len("/save"):].strip()
            if not body:
                print("  usage: /save PATH")
                continue
            lm.save(body)
            print(f"  [saved] {body}.npz + {body}.json")
            continue
        if s == "/report":
            print(f"  Ω={lm.omega:.3f}  Φ={lm.phi:.3f}  samples={lm.samples_trained}")
            continue

        _generate(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
