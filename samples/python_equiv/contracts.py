def safe_div(a, b):
    assert b != 0, "precondition: b != 0"
    r = a // b
    assert r >= 0, "postcondition: r >= 0"
    return r

assert safe_div(10, 2) == 5
assert safe_div(100, 10) == 10
print("All contract tests passed")
