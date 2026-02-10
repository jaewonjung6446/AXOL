"""
AXOL vs Python - Complexity Scaling Analysis
Which code patterns benefit AXOL most?
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
def t(src): return len(enc.encode(src.strip()))

# --- Category 1: Pure Algorithm (no data structures) ---
# Simple
algo_simple_axol = '(f fib n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))'
algo_simple_py   = 'def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)'

# Medium
algo_med_axol = '''(f gcd a b (? (= b 0) a (gcd b (% a b))))
(f lcm a b (/ (* a b) (gcd a b)))
(f is_prime n
  (? (<= n 1) false
    (D (m i 2) (m result true)
      (W (& (<= (* i i) n) result)
        (? (= (% n i) 0) (m! result false) unit)
        (m! i (+ i 1)))
      result)))'''

algo_med_py = '''def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def lcm(a, b):
    return a * b // gcd(a, b)

def is_prime(n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True'''

# Complex
algo_complex_axol = '''(f gcd a b (? (= b 0) a (gcd b (% a b))))
(f lcm a b (/ (* a b) (gcd a b)))
(f is_prime n
  (? (<= n 1) false
    (D (m i 2) (m result true)
      (W (& (<= (* i i) n) result)
        (? (= (% n i) 0) (m! result false) unit)
        (m! i (+ i 1)))
      result)))
(f sieve limit
  (m primes (A))
  (F n (range 2 limit)
    (? (is_prime n) (m! primes (append primes n)) unit))
  primes)
(f euler_totient n
  (m result n) (m p 2)
  (W (<= (* p p) n)
    (? (= (% n p) 0)
      (D (W (= (% n p) 0) (m! n (/ n p)))
        (m! result (- result (/ result p))))
      unit)
    (m! p (+ p 1)))
  (? (> n 1) (m! result (- result (/ result n))) unit)
  result)'''

algo_complex_py = '''def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def lcm(a, b):
    return a * b // gcd(a, b)

def is_prime(n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

def sieve(limit):
    primes = []
    for n in range(2, limit):
        if is_prime(n):
            primes.append(n)
    return primes

def euler_totient(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result'''

# --- Category 2: Data-Definition Heavy ---
# Simple
data_simple_axol = '(v p (S Point x 10 y 20))\n(print p.x p.y)'
data_simple_py   = 'class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\np = Point(10, 20)\nprint(p.x, p.y)'

# Medium
data_med_axol = '''(v heroes (A
  (S Unit name "Knight" hp 120 atk 30 def 20)
  (S Unit name "Mage" hp 80 atk 50 def 10)
  (S Unit name "Rogue" hp 90 atk 40 def 15)))
(v skills (H
  "slash" (S Skill dmg 25 cost 10)
  "fireball" (S Skill dmg 50 cost 30)))'''

data_med_py = '''class Unit:
    def __init__(self, name, hp, atk, defense):
        self.name = name
        self.hp = hp
        self.atk = atk
        self.defense = defense

class Skill:
    def __init__(self, dmg, cost):
        self.dmg = dmg
        self.cost = cost

heroes = [
    Unit("Knight", 120, 30, 20),
    Unit("Mage", 80, 50, 10),
    Unit("Rogue", 90, 40, 15),
]
skills = {
    "slash": Skill(25, 10),
    "fireball": Skill(50, 30),
}'''

# Complex
data_complex_axol = '''(t Stats hp mp atk def spd)
(t Equipment name slot stats)
(t Character name level stats equipment skills)

(v sword (S Equipment name "Iron Sword" slot "weapon"
  stats (S Stats hp 0 mp 0 atk 15 def 0 spd 0)))
(v shield (S Equipment name "Oak Shield" slot "offhand"
  stats (S Stats hp 0 mp 0 atk 0 def 10 spd -2)))
(v staff (S Equipment name "Oak Staff" slot "weapon"
  stats (S Stats hp 0 mp 20 atk 5 def 0 spd 0)))
(v robe (S Equipment name "Mage Robe" slot "armor"
  stats (S Stats hp 0 mp 30 atk 0 def 5 spd 0)))

(v party (A
  (S Character name "Warrior" level 5
    stats (S Stats hp 150 mp 20 atk 30 def 25 spd 10)
    equipment (A sword shield)
    skills (A "slash" "shield_bash" "taunt"))
  (S Character name "Mage" level 5
    stats (S Stats hp 80 mp 100 atk 10 def 8 spd 12)
    equipment (A staff robe)
    skills (A "fireball" "heal" "barrier"))
  (S Character name "Rogue" level 5
    stats (S Stats hp 100 mp 40 atk 25 def 12 spd 20)
    equipment (A)
    skills (A "backstab" "poison" "evade"))))'''

data_complex_py = '''class Stats:
    def __init__(self, hp, mp, atk, defense, spd):
        self.hp = hp
        self.mp = mp
        self.atk = atk
        self.defense = defense
        self.spd = spd

class Equipment:
    def __init__(self, name, slot, stats):
        self.name = name
        self.slot = slot
        self.stats = stats

class Character:
    def __init__(self, name, level, stats, equipment, skills):
        self.name = name
        self.level = level
        self.stats = stats
        self.equipment = equipment
        self.skills = skills

sword = Equipment("Iron Sword", "weapon",
    Stats(0, 0, 15, 0, 0))
shield = Equipment("Oak Shield", "offhand",
    Stats(0, 0, 0, 10, -2))
staff = Equipment("Oak Staff", "weapon",
    Stats(0, 0, 5, 0, 0))
robe = Equipment("Mage Robe", "armor",
    Stats(0, 0, 0, 5, 0))

party = [
    Character("Warrior", 5,
        Stats(150, 20, 30, 25, 10),
        [sword, shield],
        ["slash", "shield_bash", "taunt"]),
    Character("Mage", 5,
        Stats(80, 100, 10, 8, 12),
        [staff, robe],
        ["fireball", "heal", "barrier"]),
    Character("Rogue", 5,
        Stats(100, 40, 25, 12, 20),
        [],
        ["backstab", "poison", "evade"]),
]'''

# --- Category 3: Contract/Validation Heavy ---
contract_axol = '''(f transfer [i i i -> i] from_bal to_bal amount
  (Q (> amount 0))
  (Q (>= from_bal amount))
  (v new_from (- from_bal amount))
  (v new_to (+ to_bal amount))
  (G (>= new_from 0))
  (G (= (+ new_from new_to) (+ from_bal to_bal)))
  new_from)

(f withdraw [i i -> i] balance amount
  (Q (> amount 0))
  (Q (<= amount balance))
  (v result (- balance amount))
  (G (>= result 0))
  (G (= (+ result amount) balance))
  result)

(f deposit [i i -> i] balance amount
  (Q (> amount 0))
  (v result (+ balance amount))
  (G (> result balance))
  result)'''

contract_py = '''def transfer(from_bal, to_bal, amount):
    assert amount > 0, "amount must be positive"
    assert from_bal >= amount, "insufficient balance"
    new_from = from_bal - amount
    new_to = to_bal + amount
    assert new_from >= 0, "from balance must be non-negative"
    assert new_from + new_to == from_bal + to_bal, "conservation violated"
    return new_from

def withdraw(balance, amount):
    assert amount > 0, "amount must be positive"
    assert amount <= balance, "insufficient balance"
    result = balance - amount
    assert result >= 0, "result must be non-negative"
    assert result + amount == balance, "conservation violated"
    return result

def deposit(balance, amount):
    assert amount > 0, "amount must be positive"
    result = balance + amount
    assert result > balance, "deposit must increase balance"
    return result'''

# --- Category 4: HOF/Functional ---
hof_axol = '''(v nums (A 1 2 3 4 5 6 7 8 9 10))
(v evens (filter nums (L x (= (% x 2) 0))))
(v doubled (map evens (L x (* x 2))))
(v sum (reduce doubled 0 (L (acc x) (+ acc x))))
(v sorted (sort (reverse nums)))
(v found (find nums (L x (> x 7))))
(v has_neg (any nums (L x (< x 0))))
(v all_pos (all nums (L x (> x 0))))'''

hof_py = '''nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, nums))
doubled = list(map(lambda x: x * 2, evens))
total = sum(doubled)
sorted_nums = sorted(reversed(nums))
found = next((x for x in nums if x > 7), None)
has_neg = any(x < 0 for x in nums)
all_pos = all(x > 0 for x in nums)'''

# --- Run analysis ---
categories = [
    ("ALGORITHM", [
        ("simple (fib)", algo_simple_axol, algo_simple_py),
        ("medium (gcd+prime)", algo_med_axol, algo_med_py),
        ("complex (+sieve+euler)", algo_complex_axol, algo_complex_py),
    ]),
    ("DATA DEFINITION", [
        ("simple (1 struct)", data_simple_axol, data_simple_py),
        ("medium (2 structs+list)", data_med_axol, data_med_py),
        ("complex (nested RPG)", data_complex_axol, data_complex_py),
    ]),
    ("CONTRACT/VALIDATION", [
        ("3 functions w/ DbC", contract_axol, contract_py),
    ]),
    ("HOF/FUNCTIONAL", [
        ("chain of 8 HOFs", hof_axol, hof_py),
    ]),
]

print("=" * 85)
print("  AXOL vs Python - Complexity Scaling Analysis")
print("  Which code patterns benefit AXOL most?")
print("=" * 85)

for cat_name, items in categories:
    print(f"\n--- {cat_name} ---")
    print(f"{'Pattern':<28} {'AXOL':>6} {'Python':>8} {'Savings':>9} {'AXOL(c)':>9} {'Py(c)':>8} {'Char%':>8}")
    print("-" * 85)
    for name, axol, py in items:
        a = t(axol)
        p = t(py)
        ac = len(axol.strip())
        pc = len(py.strip())
        sav = (1 - a/p) * 100
        csav = (1 - ac/pc) * 100
        print(f"{name:<28} {a:>6} {p:>8} {sav:>+8.1f}% {ac:>9} {pc:>8} {csav:>+7.1f}%")

print("\n" + "=" * 85)
print("  KEY FINDINGS")
print("=" * 85)

# Compute category averages
algo_savings = []
data_savings = []
for name, axol, py in categories[0][1]:
    algo_savings.append((1 - t(axol)/t(py)) * 100)
for name, axol, py in categories[1][1]:
    data_savings.append((1 - t(axol)/t(py)) * 100)

contract_s = (1 - t(contract_axol)/t(contract_py)) * 100
hof_s = (1 - t(hof_axol)/t(hof_py)) * 100

print(f"\n  Algorithm (pure logic):    {algo_savings[0]:+.1f}% -> {algo_savings[1]:+.1f}% -> {algo_savings[2]:+.1f}%")
print(f"  Data Definition (structs): {data_savings[0]:+.1f}% -> {data_savings[1]:+.1f}% -> {data_savings[2]:+.1f}%")
print(f"  Contract/Validation:       {contract_s:+.1f}%")
print(f"  HOF/Functional chains:     {hof_s:+.1f}%")
print()
print("  Conclusion:")
print("  - Data-heavy code: AXOL advantage GROWS with complexity (class boilerplate)")
print("  - Pure algorithms: AXOL advantage is FLAT or MODEST (Python is already concise)")
print("  - Contracts/DbC: AXOL has LARGE advantage (Q/G vs assert + string)")
print("  - HOF chains: Roughly EQUAL (Python comprehensions are very concise)")
