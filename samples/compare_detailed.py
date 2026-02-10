import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def show_tokens(label, src):
    tokens = enc.encode(src)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"\n=== {label} ({len(tokens)} BPE tokens, {len(src)} chars) ===")
    print(f"Source: {src[:200]}...")
    print(f"Tokens: {decoded[:40]}{'...' if len(decoded) > 40 else ''}")
    return len(tokens), len(src)

# --- 1) 순수 알고리즘 (fibonacci) ---
print("\n" + "="*80)
print("1) 순수 알고리즘 - Fibonacci")
print("="*80)

axol1 = '(f fib [i -> i] n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))\n(F i (range 0 10) (print (fib i)))'
py1 = 'def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\nfor i in range(0, 10):\n    print(fib(i))'

a1, ac1 = show_tokens("AXOL", axol1)
p1, pc1 = show_tokens("Python", py1)

# --- 2) 계약 + 로직 ---
print("\n" + "="*80)
print("2) 계약 + 로직 - safe_div with Q/G")
print("="*80)

axol2 = '(f safe_div [i i -> i] a b (Q (!= b 0)) (v r (/ a b)) (G (>= r 0)) r)\n(! (= (safe_div 10 2) 5))\n(! (= (safe_div 100 10) 10))'
py2 = 'def safe_div(a, b):\n    assert b != 0, "precondition: b != 0"\n    r = a // b\n    assert r >= 0, "postcondition: r >= 0"\n    return r\n\nassert safe_div(10, 2) == 5\nassert safe_div(100, 10) == 10'

a2, ac2 = show_tokens("AXOL", axol2)
p2, pc2 = show_tokens("Python", py2)

# --- 3) 데이터 정의 집중 ---
print("\n" + "="*80)
print("3) 데이터 정의 집중 - struct/map 선언")
print("="*80)

axol3 = '(v u (A (S U n "K" h 120 a 30 d 20) (S U n "M" h 80 a 50 d 10) (S U n "R" h 90 a 40 d 15)))\n(v s (H "sl" (S K d 25 c 10) "fb" (S K d 50 c 30) "bs" (S K d 35 c 15)))'
py3 = 'u = [\n    {"n": "K", "h": 120, "a": 30, "d": 20},\n    {"n": "M", "h": 80, "a": 50, "d": 10},\n    {"n": "R", "h": 90, "a": 40, "d": 15},\n]\ns = {\n    "sl": {"d": 25, "c": 10},\n    "fb": {"d": 50, "c": 30},\n    "bs": {"d": 35, "c": 15},\n}'

a3, ac3 = show_tokens("AXOL", axol3)
p3, pc3 = show_tokens("Python", py3)

# --- 4) 파이프 + 람다 + 체인 ---
print("\n" + "="*80)
print("4) 파이프 + 람다 체인")
print("="*80)

axol4 = '(v r (P (range 1 20) (L xs (F x xs (? (= (% x 3) 0) (print x) unit)))))'
py4 = 'r = [print(x) for x in range(1, 20) if x % 3 == 0]'

a4, ac4 = show_tokens("AXOL", axol4)
p4, pc4 = show_tokens("Python", py4)

# --- 5) 복합 예제: 전투 시뮬레이션 ---
print("\n" + "="*80)
print("5) 복합 - 전투 시뮬레이션")
print("="*80)

axol5 = '''(f dmg [i i -> i] a d (v r (- a d)) (? (< r 0) 0 r))
(f hit [i i i -> i] hp a d (v h (- hp (dmg a d))) (? (< h 0) 0 h))
(f sim [i i i i i i -> s] h1 a1 d1 h2 a2 d2
  (m hp1 h1) (m hp2 h2) (m t 0)
  (W (& (> hp1 0) (> hp2 0))
    (m! t (+ t 1))
    (? (= (% t 2) 1)
      (m! hp2 (hit hp2 a1 d2))
      (m! hp1 (hit hp1 a2 d1))))
  (? (> hp1 0) "p1" "p2"))
(print (sim 100 25 10 80 30 8))'''

py5 = '''def dmg(a, d):
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

a5, ac5 = show_tokens("AXOL", axol5)
p5, pc5 = show_tokens("Python", py5)

# --- 총합 ---
print("\n" + "="*80)
print("종합 결과")
print("="*80)

results = [
    ("1.fibonacci", a1, p1, ac1, pc1),
    ("2.contracts", a2, p2, ac2, pc2),
    ("3.data_def",  a3, p3, ac3, pc3),
    ("4.pipe+lambda", a4, p4, ac4, pc4),
    ("5.combat_sim", a5, p5, ac5, pc5),
]

print(f"\n{'예제':<16} {'AXOL':<8} {'Python':<9} {'BPE절감':<10} {'AXOL(c)':<10} {'Py(c)':<10} {'문자절감'}")
print("-" * 75)

ta = tp = tac = tpc = 0
for name, a, p, ac, pc in results:
    bpe_save = (1 - a/p) * 100
    c_save = (1 - ac/pc) * 100
    ta += a; tp += p; tac += ac; tpc += pc
    print(f"{name:<16} {a:<8} {p:<9} {bpe_save:>+6.1f}%    {ac:<10} {pc:<10} {c_save:>+6.1f}%")

print("-" * 75)
print(f"{'합계':<16} {ta:<8} {tp:<9} {(1-ta/tp)*100:>+6.1f}%    {tac:<10} {tpc:<10} {(1-tac/tpc)*100:>+6.1f}%")
