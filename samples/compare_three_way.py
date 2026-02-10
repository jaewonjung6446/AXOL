"""
AXOL vs Python vs C# - BPE Token & Character Comparison
Uses tiktoken cl100k_base (GPT-4 / Claude tokenizer approximation)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tiktoken
import os

enc = tiktoken.get_encoding("cl100k_base")

def count(src):
    tokens = len(enc.encode(src.strip()))
    chars = len(src.strip())
    return tokens, chars

base = os.path.dirname(os.path.abspath(__file__))

pairs = [
    ("fibonacci",  "fibonacci.axol",  "python_equiv/fibonacci.py",  "csharp_equiv/fibonacci.cs"),
    ("contracts",  "contracts.axol",  "python_equiv/contracts.py",  "csharp_equiv/contracts.cs"),
    ("combat",     "combat.axol",     "python_equiv/combat.py",     "csharp_equiv/combat.cs"),
    ("data_heavy", "data_heavy.axol", "python_equiv/data_heavy.py", "csharp_equiv/data_heavy.cs"),
]

print("=" * 110)
print("AXOL vs Python vs C# - BPE Token Comparison (cl100k_base)")
print("=" * 110)

header = f"{'Example':<14} {'AXOL':>6} {'Python':>8} {'C#':>6} {'vs Py':>8} {'vs C#':>8} {'AXOL(c)':>9} {'Py(c)':>8} {'C#(c)':>7} {'c vs Py':>9} {'c vs C#':>9}"
print(header)
print("-" * 110)

ta = tp = tc = tac = tpc = tcc = 0

results = []
for name, axol_path, py_path, cs_path in pairs:
    axol_src = open(os.path.join(base, axol_path), encoding="utf-8").read()
    py_src   = open(os.path.join(base, py_path),   encoding="utf-8").read()
    cs_src   = open(os.path.join(base, cs_path),    encoding="utf-8").read()

    a_tok, a_chr = count(axol_src)
    p_tok, p_chr = count(py_src)
    c_tok, c_chr = count(cs_src)

    vs_py = (1 - a_tok / p_tok) * 100
    vs_cs = (1 - a_tok / c_tok) * 100
    c_vs_py = (1 - a_chr / p_chr) * 100
    c_vs_cs = (1 - a_chr / c_chr) * 100

    ta += a_tok; tp += p_tok; tc += c_tok
    tac += a_chr; tpc += p_chr; tcc += c_chr

    print(f"{name:<14} {a_tok:>6} {p_tok:>8} {c_tok:>6} {vs_py:>+7.1f}% {vs_cs:>+7.1f}% {a_chr:>9} {p_chr:>8} {c_chr:>7} {c_vs_py:>+8.1f}% {c_vs_cs:>+8.1f}%")
    results.append((name, a_tok, p_tok, c_tok, a_chr, p_chr, c_chr))

print("-" * 110)
tot_vs_py = (1 - ta / tp) * 100
tot_vs_cs = (1 - ta / tc) * 100
tot_c_py = (1 - tac / tpc) * 100
tot_c_cs = (1 - tac / tcc) * 100
print(f"{'TOTAL':<14} {ta:>6} {tp:>8} {tc:>6} {tot_vs_py:>+7.1f}% {tot_vs_cs:>+7.1f}% {tac:>9} {tpc:>8} {tcc:>7} {tot_c_py:>+8.1f}% {tot_c_cs:>+8.1f}%")

print()
print("=" * 110)
print("SUMMARY")
print("=" * 110)
print(f"  AXOL uses {tot_vs_py:+.1f}% BPE tokens vs Python  ({ta} vs {tp})")
print(f"  AXOL uses {tot_vs_cs:+.1f}% BPE tokens vs C#      ({ta} vs {tc})")
print(f"  AXOL uses {tot_c_py:+.1f}% characters vs Python   ({tac} vs {tpc})")
print(f"  AXOL uses {tot_c_cs:+.1f}% characters vs C#       ({tac} vs {tcc})")

# --- Markdown table for README ---
print()
print("=" * 110)
print("MARKDOWN TABLE (for README.md)")
print("=" * 110)
print()
print("| Example | AXOL | Python | C# | vs Python | vs C# |")
print("|---------|------|--------|----|-----------|-------|")
for name, a, p, c, ac, pc, cc in results:
    vs_py = (1 - a / p) * 100
    vs_cs = (1 - a / c) * 100
    print(f"| {name} | {a} | {p} | {c} | {vs_py:+.1f}% | {vs_cs:+.1f}% |")
print(f"| **Total** | **{ta}** | **{tp}** | **{tc}** | **{tot_vs_py:+.1f}%** | **{tot_vs_cs:+.1f}%** |")
