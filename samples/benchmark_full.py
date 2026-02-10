"""
AXOL vs Python vs C# - Full BPE Token & Character Benchmark
Uses tiktoken cl100k_base (GPT-4 / Claude tokenizer approximation)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tiktoken
import os

enc = tiktoken.get_encoding("cl100k_base")
base = os.path.dirname(os.path.abspath(__file__))

def count(src):
    s = src.strip()
    return len(enc.encode(s)), len(s)

def read(path):
    return open(os.path.join(base, path), encoding="utf-8").read()

# --- Phase 1 samples (original syntax) ---
phase1 = [
    ("fibonacci",  "fibonacci.axol",  "python_equiv/fibonacci.py",  "csharp_equiv/fibonacci.cs"),
    ("contracts",  "contracts.axol",  "python_equiv/contracts.py",  "csharp_equiv/contracts.cs"),
    ("combat",     "combat.axol",     "python_equiv/combat.py",     "csharp_equiv/combat.cs"),
    ("data_heavy", "data_heavy.axol", "python_equiv/data_heavy.py", "csharp_equiv/data_heavy.cs"),
]

# --- Phase 2 samples (enhanced syntax) ---
phase2 = [
    ("pattern+module", "phase2/pattern_match.axol", "python_equiv/pattern_match.py", "csharp_equiv/pattern_match.cs"),
]

# --- Inline combat sim (from compare_detailed.py) ---
inline_axol = '''(f dmg a d (v r (- a d)) (? (< r 0) 0 r))
(f hit hp a d (v h (- hp (dmg a d))) (? (< h 0) 0 h))
(f sim h1 a1 d1 h2 a2 d2
  (m hp1 h1) (m hp2 h2) (m t 0)
  (W (& (> hp1 0) (> hp2 0))
    (m! t (+ t 1))
    (? (= (% t 2) 1)
      (m! hp2 (hit hp2 a1 d2))
      (m! hp1 (hit hp1 a2 d1))))
  (? (> hp1 0) "p1" "p2"))
(print (sim 100 25 10 80 30 8))'''

inline_py = '''def dmg(a, d):
    r = a - d
    return 0 if r < 0 else r

def hit(hp, a, d):
    h = hp - dmg(a, d)
    return 0 if h < 0 else h

def sim(h1, a1, d1, h2, a2, d2):
    hp1, hp2, t = h1, h2, 0
    while hp1 > 0 and hp2 > 0:
        t += 1
        if t % 2 == 1:
            hp2 = hit(hp2, a1, d2)
        else:
            hp1 = hit(hp1, a2, d1)
    return "p1" if hp1 > 0 else "p2"

print(sim(100, 25, 10, 80, 30, 8))'''

inline_cs = '''using System;

class Program
{
    static int Dmg(int a, int d)
    {
        int r = a - d;
        return r < 0 ? 0 : r;
    }

    static int Hit(int hp, int a, int d)
    {
        int h = hp - Dmg(a, d);
        return h < 0 ? 0 : h;
    }

    static string Sim(int h1, int a1, int d1, int h2, int a2, int d2)
    {
        int hp1 = h1, hp2 = h2, t = 0;
        while (hp1 > 0 && hp2 > 0)
        {
            t++;
            if (t % 2 == 1)
                hp2 = Hit(hp2, a1, d2);
            else
                hp1 = Hit(hp1, a2, d1);
        }
        return hp1 > 0 ? "p1" : "p2";
    }

    static void Main()
    {
        Console.WriteLine(Sim(100, 25, 10, 80, 30, 8));
    }
}'''

print("=" * 100)
print("  AXOL BPE Token Efficiency Benchmark")
print("  Tokenizer: tiktoken cl100k_base (GPT-4 / Claude approximation)")
print("=" * 100)

all_results = []

def run_section(title, items, inline_data=None):
    global all_results
    print(f"\n--- {title} ---")
    print(f"{'Example':<18} {'AXOL':>6} {'Python':>8} {'C#':>6} {'vs Py':>8} {'vs C#':>8}")
    print("-" * 60)

    section_results = []
    for name, axol_p, py_p, cs_p in items:
        a_src = read(axol_p)
        p_src = read(py_p)
        c_src = read(cs_p)
        a_tok, a_chr = count(a_src)
        p_tok, p_chr = count(p_src)
        c_tok, c_chr = count(c_src)
        vs_py = (1 - a_tok / p_tok) * 100
        vs_cs = (1 - a_tok / c_tok) * 100
        print(f"{name:<18} {a_tok:>6} {p_tok:>8} {c_tok:>6} {vs_py:>+7.1f}% {vs_cs:>+7.1f}%")
        section_results.append((name, a_tok, p_tok, c_tok, a_chr, p_chr, c_chr))

    if inline_data:
        for name, a_src, p_src, c_src in inline_data:
            a_tok, a_chr = count(a_src)
            p_tok, p_chr = count(p_src)
            c_tok, c_chr = count(c_src)
            vs_py = (1 - a_tok / p_tok) * 100
            vs_cs = (1 - a_tok / c_tok) * 100
            print(f"{name:<18} {a_tok:>6} {p_tok:>8} {c_tok:>6} {vs_py:>+7.1f}% {vs_cs:>+7.1f}%")
            section_results.append((name, a_tok, p_tok, c_tok, a_chr, p_chr, c_chr))

    all_results.extend(section_results)
    return section_results

run_section("Phase 1 Samples (S-expression syntax)", phase1,
            inline_data=[("combat_sim_inline", inline_axol, inline_py, inline_cs)])
run_section("Phase 2 Samples (enhanced syntax)", phase2)

# --- Grand total ---
ta = sum(r[1] for r in all_results)
tp = sum(r[2] for r in all_results)
tc = sum(r[3] for r in all_results)
tac = sum(r[4] for r in all_results)
tpc = sum(r[5] for r in all_results)
tcc = sum(r[6] for r in all_results)

print("\n" + "=" * 100)
print("  GRAND TOTAL")
print("=" * 100)
print(f"  Total BPE Tokens:  AXOL={ta}  Python={tp}  C#={tc}")
print(f"  vs Python:  {(1-ta/tp)*100:+.1f}% BPE tokens,  {(1-tac/tpc)*100:+.1f}% characters")
print(f"  vs C#:      {(1-ta/tc)*100:+.1f}% BPE tokens,  {(1-tac/tcc)*100:+.1f}% characters")
print()

# --- README markdown ---
print("=" * 100)
print("  MARKDOWN for README")
print("=" * 100)
print()
print("| Example | AXOL (tokens) | Python (tokens) | C# (tokens) | vs Python | vs C# |")
print("|---------|:---:|:---:|:---:|:---:|:---:|")
for name, a, p, c, ac, pc, cc in all_results:
    vs_py = (1 - a / p) * 100
    vs_cs = (1 - a / c) * 100
    print(f"| {name} | {a} | {p} | {c} | **{vs_py:+.1f}%** | **{vs_cs:+.1f}%** |")
print(f"| **Total** | **{ta}** | **{tp}** | **{tc}** | **{(1-ta/tp)*100:+.1f}%** | **{(1-ta/tc)*100:+.1f}%** |")
