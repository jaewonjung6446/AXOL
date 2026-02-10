"""
Token Optimization Benchmark
Measures savings from the 3 new optimizations:
1. Indentation mode (.axoli - no closing parens)
2. Short enum aliases (.Variant)
3. Positional struct fields
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
def t(src): return len(enc.encode(src.strip()))
def show(label, src):
    tokens = t(src)
    chars = len(src.strip())
    print(f"  {label:<45} {tokens:>4} tokens  ({chars:>4} chars)")
    return tokens

print("=" * 80)
print("  AXOL Token Optimization Benchmark")
print("  Measuring actual savings from Phase 2 optimizations")
print("=" * 80)

# ============================================================
# 1. INDENTATION MODE (.axoli)
# ============================================================
print("\n--- 1. Indentation Mode (.axoli) ---")
print("    Eliminates closing parentheses via indentation\n")

# Example A: Simple function
standard_a = '(f fib n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))'
indent_a = '''f fib n
  ? (<= n 1) n
    + (fib (- n 1)) (fib (- n 2))'''

sa = show("Standard AXOL", standard_a)
ia = show("Indent mode (.axoli)", indent_a)
print(f"  Savings: {(1-ia/sa)*100:+.1f}%\n")

# Example B: Combat function
standard_b = '''(f apply_dmg ent amt
  (Q (>= amt 0))
  (v oh (@ ent hp))
  (v raw (- amt (@ ent def)))
  (v dmg (? (< raw 0) 0 raw))
  (m! (@ ent hp) (max 0 (- oh dmg)))
  (G (<= (@ ent hp) oh))
  ent)'''

indent_b = '''f apply_dmg ent amt
  Q (>= amt 0)
  v oh (@ ent hp)
  v raw (- amt (@ ent def))
  v dmg (? (< raw 0) 0 raw)
  m! (@ ent hp) (max 0 (- oh dmg))
  G (<= (@ ent hp) oh)
  ent'''

sb = show("Standard AXOL (combat)", standard_b)
ib = show("Indent mode (combat)", indent_b)
print(f"  Savings: {(1-ib/sb)*100:+.1f}%\n")

# Example C: Data definitions
standard_c = '''(t Stats hp mp atk def spd)
(v warrior (S Character name "Warrior" level 5
  stats (S Stats hp 150 mp 20 atk 30 def 25 spd 10)
  skills (A "slash" "bash")))
(v mage (S Character name "Mage" level 5
  stats (S Stats hp 80 mp 100 atk 10 def 8 spd 12)
  skills (A "fire" "heal")))'''

indent_c = '''t Stats hp mp atk def spd
v warrior (S Character name "Warrior" level 5
  stats (S Stats hp 150 mp 20 atk 30 def 25 spd 10)
  skills (A "slash" "bash"))
v mage (S Character name "Mage" level 5
  stats (S Stats hp 80 mp 100 atk 10 def 8 spd 12)
  skills (A "fire" "heal"))'''

sc = show("Standard AXOL (data)", standard_c)
ic = show("Indent mode (data)", indent_c)
print(f"  Savings: {(1-ic/sc)*100:+.1f}%")

indent_avg = (1 - (ia+ib+ic)/(sa+sb+sc)) * 100

# ============================================================
# 2. SHORT ENUM ALIASES (.Variant)
# ============================================================
print("\n--- 2. Short Enum Aliases (.Variant) ---")
print("    Use .Variant instead of EnumName.Variant in match\n")

# Example A: Simple enum match
full_a = '''(e Color Red Green Blue)
(v c Color.Green)
(X c Color.Red "red" Color.Green "green" Color.Blue "blue" _ "unknown")'''

short_a = '''(e Color Red Green Blue)
(v c Color.Green)
(X c .Red "red" .Green "green" .Blue "blue" _ "unknown")'''

fa = show("Full enum names", full_a)
sha = show("Short aliases (.Variant)", short_a)
print(f"  Savings: {(1-sha/fa)*100:+.1f}%\n")

# Example B: Data-bearing enum with match
full_b = '''(e Shape (Circle r) (Rect w h) (Triangle b h))
(f area shape
  (X shape
    (Shape.Circle r) (* 3.14 (* r r))
    (Shape.Rect w h) (* w h)
    (Shape.Triangle b h) (* 0.5 (* b h))))
(f describe shape
  (X shape
    (Shape.Circle r) (format "circle r={}" r)
    (Shape.Rect w h) (format "rect {}x{}" w h)
    (Shape.Triangle b h) (format "tri b={} h={}" b h)))'''

short_b = '''(e Shape (Circle r) (Rect w h) (Triangle b h))
(f area shape
  (X shape
    (.Circle r) (* 3.14 (* r r))
    (.Rect w h) (* w h)
    (.Triangle b h) (* 0.5 (* b h))))
(f describe shape
  (X shape
    (.Circle r) (format "circle r={}" r)
    (.Rect w h) (format "rect {}x{}" w h)
    (.Triangle b h) (format "tri b={} h={}" b h)))'''

fb = show("Full enum names (shapes)", full_b)
shb = show("Short aliases (shapes)", short_b)
print(f"  Savings: {(1-shb/fb)*100:+.1f}%")

enum_avg = (1 - (sha+shb)/(fa+fb)) * 100

# ============================================================
# 3. POSITIONAL STRUCT FIELDS
# ============================================================
print("\n--- 3. Positional Struct Fields ---")
print("    Use (S Point 10 20) instead of (S Point x 10 y 20)\n")

# Example A: Simple struct
named_a = '''(t Point x y)
(v p (S Point x 10 y 20))
(v q (S Point x 30 y 40))
(+ (@ p x) (@ q y))'''

positional_a = '''(t Point x y)
(v p (S Point 10 20))
(v q (S Point 30 40))
(+ (@ p x) (@ q y))'''

na = show("Named fields", named_a)
pa = show("Positional fields", positional_a)
print(f"  Savings: {(1-pa/na)*100:+.1f}%\n")

# Example B: Game entity definition
named_b = '''(t Stats hp mp atk def spd)
(t Equipment name slot bonus)
(v sword (S Equipment name "Iron Sword" slot "weapon" bonus 15))
(v shield (S Equipment name "Oak Shield" slot "offhand" bonus 10))
(v staff (S Equipment name "Oak Staff" slot "weapon" bonus 5))
(v warrior_stats (S Stats hp 150 mp 20 atk 30 def 25 spd 10))
(v mage_stats (S Stats hp 80 mp 100 atk 10 def 8 spd 12))
(v rogue_stats (S Stats hp 100 mp 40 atk 25 def 12 spd 20))'''

positional_b = '''(t Stats hp mp atk def spd)
(t Equipment name slot bonus)
(v sword (S Equipment "Iron Sword" "weapon" 15))
(v shield (S Equipment "Oak Shield" "offhand" 10))
(v staff (S Equipment "Oak Staff" "weapon" 5))
(v warrior_stats (S Stats 150 20 30 25 10))
(v mage_stats (S Stats 80 100 10 8 12))
(v rogue_stats (S Stats 100 40 25 12 20))'''

nb = show("Named fields (game entities)", named_b)
pb = show("Positional fields (game entities)", positional_b)
print(f"  Savings: {(1-pb/nb)*100:+.1f}%")

pos_avg = (1 - (pa+pb)/(na+nb)) * 100

# ============================================================
# 4. COMBINED: All 3 optimizations together
# ============================================================
print("\n--- 4. Combined Optimization (All 3 Together) ---")
print("    Indent mode + short enums + positional structs\n")

original = '''(t Stats hp mp atk def spd)
(e Shape (Circle r) (Rect w h))
(v warrior (S Stats hp 150 mp 20 atk 30 def 25 spd 10))
(v mage (S Stats hp 80 mp 100 atk 10 def 8 spd 12))
(f area shape
  (X shape
    (Shape.Circle r) (* 3.14 (* r r))
    (Shape.Rect w h) (* w h)))
(f total_hp units
  (reduce units 0 (L (acc u) (+ acc (@ u hp)))))'''

optimized = '''t Stats hp mp atk def spd
e Shape (Circle r) (Rect w h)
v warrior (S Stats 150 20 30 25 10)
v mage (S Stats 80 100 10 8 12)
f area shape
  X shape
    (.Circle r) (* 3.14 (* r r))
    (.Rect w h) (* w h)
f total_hp units
  reduce units 0 (L (acc u) (+ acc (@ u hp)))'''

o = show("Original AXOL", original)
opt = show("Fully optimized (.axoli + short + pos)", optimized)
combined_savings = (1-opt/o)*100
print(f"  Combined savings: {combined_savings:+.1f}%")

# ============================================================
# 5. vs Python (with all optimizations)
# ============================================================
print("\n--- 5. Optimized AXOL vs Python ---\n")

python_equiv = '''class Stats:
    def __init__(self, hp, mp, atk, defense, spd):
        self.hp = hp
        self.mp = mp
        self.atk = atk
        self.defense = defense
        self.spd = spd

class Circle:
    def __init__(self, r):
        self.r = r

class Rect:
    def __init__(self, w, h):
        self.w = w
        self.h = h

warrior = Stats(150, 20, 30, 25, 10)
mage = Stats(80, 100, 10, 8, 12)

def area(shape):
    if isinstance(shape, Circle):
        return 3.14 * shape.r * shape.r
    elif isinstance(shape, Rect):
        return shape.w * shape.h

from functools import reduce
def total_hp(units):
    return reduce(lambda acc, u: acc + u.hp, units, 0)'''

py = show("Python equivalent", python_equiv)
show("AXOL (standard)", original)
show("AXOL (fully optimized)", optimized)
print(f"\n  AXOL standard vs Python:   {(1-o/py)*100:+.1f}%")
print(f"  AXOL optimized vs Python:  {(1-opt/py)*100:+.1f}%")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("  OPTIMIZATION SUMMARY")
print("=" * 80)
print(f"\n  {'Optimization':<40} {'Avg Savings':>15}")
print(f"  {'-'*40} {'-'*15}")
print(f"  {'1. Indentation mode (.axoli)':<40} {indent_avg:>+14.1f}%")
print(f"  {'2. Short enum aliases (.Variant)':<40} {enum_avg:>+14.1f}%")
print(f"  {'3. Positional struct fields':<40} {pos_avg:>+14.1f}%")
print(f"  {'ALL COMBINED':<40} {combined_savings:>+14.1f}%")
print(f"\n  vs Python (standard AXOL):  {(1-o/py)*100:+.1f}%")
print(f"  vs Python (optimized AXOL): {(1-opt/py)*100:+.1f}%")
print()
