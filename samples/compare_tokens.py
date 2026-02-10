import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude tokenizer 근사

pairs = [
    ("fibonacci",   "fibonacci.axol",              "python_equiv/fibonacci.py"),
    ("contracts",   "contracts.axol",              "python_equiv/contracts.py"),
    ("combat",      "combat.axol",                 "python_equiv/combat.py"),
    ("data_heavy",  "data_heavy.axol",             "python_equiv/data_heavy.py"),
]

print(f"{'예제':<14} {'AXOL(BPE)':<12} {'Python(BPE)':<13} {'절감률':<10} {'AXOL(chars)':<13} {'Python(chars)':<14} {'문자 절감'}")
print("-" * 95)

total_axol = 0
total_py = 0
total_axol_c = 0
total_py_c = 0

for name, axol_path, py_path in pairs:
    axol_src = open(axol_path, encoding="utf-8").read().strip()
    py_src = open(py_path, encoding="utf-8").read().strip()

    axol_tokens = len(enc.encode(axol_src))
    py_tokens = len(enc.encode(py_src))

    saving = (1 - axol_tokens / py_tokens) * 100
    char_saving = (1 - len(axol_src) / len(py_src)) * 100

    total_axol += axol_tokens
    total_py += py_tokens
    total_axol_c += len(axol_src)
    total_py_c += len(py_src)

    print(f"{name:<14} {axol_tokens:<12} {py_tokens:<13} {saving:>+6.1f}%    {len(axol_src):<13} {len(py_src):<14} {char_saving:>+6.1f}%")

print("-" * 95)
total_saving = (1 - total_axol / total_py) * 100
total_char_saving = (1 - total_axol_c / total_py_c) * 100
print(f"{'합계':<14} {total_axol:<12} {total_py:<13} {total_saving:>+6.1f}%    {total_axol_c:<13} {total_py_c:<14} {total_char_saving:>+6.1f}%")
