"""
Token Optimization Research
Exploring techniques beyond current AXOL syntax
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
def t(src): return len(enc.encode(src.strip()))
def show(label, src):
    tokens = t(src)
    print(f"  {label:<40} {tokens:>4} tokens  ({len(src.strip()):>4} chars)")
    return tokens

print("=" * 80)
print("  Token Optimization Research")
print("=" * 80)

# === 1. Indentation-based S-expressions (eliminating closing parens) ===
print("\n--- 1. Closing Paren Elimination (Indentation-based) ---")
print("    Idea: Use indentation instead of closing parentheses\n")

current = '''(f apply_dmg ent amt
  (Q (>= amt 0))
  (v oh (@ ent hp))
  (v raw (- amt (@ ent def)))
  (v dmg (? (< raw 0) 0 raw))
  (m! (@ ent hp) (max 0 (- oh dmg)))
  (G (<= (@ ent hp) oh))
  ent)'''

indented = '''f apply_dmg ent amt
  Q (>= amt 0)
  v oh ent.hp
  v raw (- amt ent.def)
  v dmg (? (< raw 0) 0 raw)
  m! ent.hp (max 0 (- oh dmg))
  G (<= ent.hp oh)
  ent'''

a = show("Current AXOL", current)
b = show("Indentation-based (no close parens)", indented)
print(f"  Savings: {(1-b/a)*100:+.1f}%")

# === 2. Implicit Arity (drop parens for known-arity builtins) ===
print("\n--- 2. Implicit Arity (known builtins skip parens) ---")
print("    Idea: + - * / are always binary, don't need wrapping parens\n")

current2 = '(v result (+ (* a b) (- c (/ d e))))'
implicit2 = 'v result + * a b - c / d e'

a = show("Current AXOL", current2)
b = show("Implicit arity (postfix/prefix)", implicit2)
print(f"  Savings: {(1-b/a)*100:+.1f}%")

# === 3. Template/Macro System ===
print("\n--- 3. Template/Macro System ---")
print("    Idea: Define reusable patterns, reference by short name\n")

no_macro = '''(v warrior (S Character name "Warrior" hp 150 mp 20 atk 30 def 25 spd 10))
(v mage (S Character name "Mage" hp 80 mp 100 atk 10 def 8 spd 12))
(v rogue (S Character name "Rogue" hp 100 mp 40 atk 25 def 12 spd 20))
(v healer (S Character name "Healer" hp 90 mp 120 atk 8 def 10 spd 11))'''

with_macro = '''(T C name hp mp atk def spd = (S Character name $1 hp $2 mp $3 atk $4 def $5 spd $6))
(v warrior (C "Warrior" 150 20 30 25 10))
(v mage (C "Mage" 80 100 10 8 12))
(v rogue (C "Rogue" 100 40 25 12 20))
(v healer (C "Healer" 90 120 8 10 11))'''

a = show("Without macros (repetitive structs)", no_macro)
b = show("With macro template", with_macro)
print(f"  Savings: {(1-b/a)*100:+.1f}%")

# === 4. Semantic Compression (Reference dedup) ===
print("\n--- 4. Structural Deduplication ---")
print("    Idea: Assign IDs to repeated sub-expressions\n")

repeated = '''(f area shape
  (X shape
    (Shape.Circle r) (* 3.14 (* r r))
    (Shape.Rect w h) (* w h)
    (Shape.Triangle b h) (* 0.5 (* b h))))
(f perimeter shape
  (X shape
    (Shape.Circle r) (* 6.28 r)
    (Shape.Rect w h) (* 2 (+ w h))
    (Shape.Triangle a b c) (+ a (+ b c))))'''

deduped = '''(f area s (X s .Ci r (* 3.14 (* r r)) .Re w h (* w h) .Tr b h (* 0.5 (* b h))))
(f perimeter s (X s .Ci r (* 6.28 r) .Re w h (* 2 (+ w h)) .Tr a b c (+ a (+ b c))))'''

a = show("Verbose enum matching", repeated)
b = show("Short enum aliases (.Ci .Re .Tr)", deduped)
print(f"  Savings: {(1-b/a)*100:+.1f}%")

# === 5. System Prompt Convention ===
print("\n--- 5. System Prompt Convention ---")
print("    Idea: Define abbreviations in system prompt, use in code\n")

system_prompt = '''You generate AXOL code. Convention: when defining game entities,
use these abbreviations: C=Character, W=Weapon, S=Skill, hp/mp/atk/def/spd as fields.'''

normal_code = '''(v hero (S Character name "Hero" hp 100 atk 25 def 10))
(v sword (S Weapon name "Iron Sword" damage 15 speed 10))
(v skill (S Skill name "Slash" power 20 cost 5))'''

# With convention the LLM would know to use short forms
convention_code = '''(v hero (S C n "Hero" h 100 a 25 d 10))
(v sword (S W n "Iron Sword" dm 15 sp 10))
(v skill (S K n "Slash" pw 20 co 5))'''

prompt_cost = t(system_prompt)
a = show("Normal code (no convention)", normal_code)
b = show("With abbreviation convention", convention_code)
print(f"  System prompt overhead: {prompt_cost} tokens (one-time)")
print(f"  Per-generation savings: {(1-b/a)*100:+.1f}%")
print(f"  Break-even after: {prompt_cost / (a - b):.0f} generations")

# === 6. Binary/Encoded Representation ===
print("\n--- 6. Tokenizer Vocabulary Analysis ---")
print("    Idea: Choose keywords that are single tokens in cl100k_base\n")

# Check which single characters are single tokens
single_char_tokens = []
multi_char_single_tokens = []

for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    toks = enc.encode(c)
    if len(toks) == 1:
        single_char_tokens.append(c)

# Check common keywords
keywords = ['def', 'return', 'if', 'else', 'for', 'while', 'class', 'import',
            'fn', 'let', 'var', 'val', 'mut', 'pub', 'mod', 'use',
            'match', 'case', 'enum', 'struct', 'type', 'trait']
for kw in keywords:
    toks = enc.encode(kw)
    if len(toks) == 1:
        multi_char_single_tokens.append((kw, toks[0]))

print(f"  Single chars that are 1 token: {len(single_char_tokens)}/52")
print(f"  Multi-char words that are 1 token: {[k for k,_ in multi_char_single_tokens]}")

# Check AXOL keywords
axol_kws = ['f', 'v', 'L', 'P', 'F', 'W', 'D', 'S', 'A', 'H', 'M', 'X', 'Q', 'G', 'e', 't']
print(f"\n  AXOL keywords token count:")
for kw in axol_kws:
    toks = enc.encode(kw)
    print(f"    '{kw}' -> {len(toks)} token(s)")

# === 7. Newline vs Space as delimiter ===
print("\n--- 7. Delimiter Analysis ---")
print("    Idea: How do different delimiters tokenize?\n")

for delim_name, delim_src in [
    ("space-separated", "(v a 1) (v b 2) (v c 3)"),
    ("newline-separated", "(v a 1)\n(v b 2)\n(v c 3)"),
    ("semicolon-separated", "(v a 1);(v b 2);(v c 3)"),
]:
    show(delim_name, delim_src)

# === Summary ===
print("\n" + "=" * 80)
print("  OPTIMIZATION POTENTIAL SUMMARY")
print("=" * 80)

techniques = [
    ("Indentation-based (no close parens)", "-15~25%", "Medium", "Parser rewrite needed"),
    ("Implicit arity for builtins", "-10~20%", "Medium", "Ambiguity risk"),
    ("Template/Macro system", "-20~40%", "High (repetitive code)", "New language feature"),
    ("Short enum aliases", "-10~15%", "Medium", "Convention only"),
    ("System prompt conventions", "-15~25%", "High (amortized)", "No code change needed"),
    ("Tokenizer-aligned keywords", "Already optimal", "N/A", "AXOL already uses 1-token chars"),
]

print(f"\n{'Technique':<40} {'Savings':<15} {'Impact':<25} {'Effort':<30}")
print("-" * 110)
for name, savings, impact, effort in techniques:
    print(f"{name:<40} {savings:<15} {impact:<25} {effort:<30}")
