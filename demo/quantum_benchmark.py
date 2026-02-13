"""
Quantum Computing Benchmark: AXOL vs Quantum Simulator (numpy)

Same mathematical operations, different implementations:
  - AXOL: Rust-native chaos-theory engine
  - Quantum Sim: numpy-based statevector simulator
  - (Reference: theoretical quantum hardware)

Operations tested:
  [1] Born rule measurement: |psi|^2 -> probabilities
  [2] Density matrix construction: rho = |psi><psi| + purity Tr(rho^2)
  [3] Interference: psi = alpha*|psi1> + beta*|psi2>
  [4] Dephasing channel: Kraus operators
  [5] Von Neumann entropy: -Tr(rho * log(rho))
"""
import numpy as np
import time
import subprocess
import os
import sys

AXOL_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "axol-lang", "target", "release", "axol.exe")

ITERATIONS = 100_000

# =====================================================================
# Quantum Simulator (pure numpy)
# =====================================================================

def make_state(dim):
    """Create a normalized complex state vector."""
    psi = np.array([np.sin(i) + 1j * np.cos(i) for i in range(dim)], dtype=np.complex128)
    return psi / np.linalg.norm(psi)


def born_rule(psi):
    """Born rule: P(i) = |psi_i|^2"""
    return np.abs(psi) ** 2


def density_matrix(psi):
    """rho = |psi><psi|"""
    return np.outer(psi, np.conj(psi))


def purity(rho):
    """Tr(rho^2)"""
    return np.real(np.trace(rho @ rho))


def interfere(psi1, psi2, alpha=0.7):
    """Quantum interference: psi = alpha*psi1 + (1-alpha)*psi2, normalized."""
    psi = alpha * psi1 + (1 - alpha) * psi2
    norm = np.linalg.norm(psi)
    if norm > 1e-15:
        psi /= norm
    return psi


def dephasing_kraus(gamma, dim):
    """Dephasing channel Kraus operators."""
    ops = [np.sqrt(1 - gamma) * np.eye(dim, dtype=np.complex128)]
    for k in range(dim):
        E = np.zeros((dim, dim), dtype=np.complex128)
        E[k, k] = np.sqrt(gamma / dim)
        ops.append(E)
    return ops


def apply_channel(rho, kraus_ops):
    """Apply quantum channel: rho' = sum_k E_k rho E_k^dagger"""
    result = np.zeros_like(rho)
    for E in kraus_ops:
        result += E @ rho @ E.conj().T
    return result


def von_neumann_entropy(rho):
    """S = -Tr(rho * log(rho))"""
    eigenvalues = np.real(np.linalg.eigvalsh(rho))
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log(eigenvalues))


# =====================================================================
# Benchmark Runner
# =====================================================================

def bench_quantum_sim(name, func, iterations, *args):
    """Benchmark a quantum simulator function."""
    # Warmup
    for _ in range(min(1000, iterations // 10)):
        func(*args)

    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    elapsed = time.perf_counter() - start

    us_per_op = (elapsed / iterations) * 1e6
    return us_per_op


def get_axol_results():
    """Pre-measured AXOL benchmark results (us per operation)."""
    return {
        "Born rule (measure)": {8: 0.118, 64: 0.194, 256: 0.525, 1024: 1.852},
        "Density matrix (from_pure + purity)": {4: 0.141, 8: 0.432, 16: 0.855, 32: 3.008},
        "Interference": {8: 0.134, 64: 0.358, 256: 1.184, 1024: 4.904},
        "Dephasing channel (apply)": {4: 1.354, 8: 5.423, 16: 40.171, 32: 256.081},
        "Von Neumann entropy": {4: 5.052, 8: 13.095, 16: 42.808},
    }


def main():
    print("#" * 65)
    print("#  QUANTUM BENCHMARK: AXOL (Rust) vs Quantum Simulator (numpy)")
    print("#  Same math. Different implementation.")
    print("#" * 65)
    print()

    # ── AXOL results (pre-measured from `axol bench`) ──
    print("  Loading AXOL benchmark results (pre-measured Rust native)...")
    axol = get_axol_results()
    print("  Done.")
    print()

    # ── Test dimensions ──
    dims_small = [4, 8, 16, 32]
    dims_large = [8, 64, 256, 1024]

    # =================================================================
    # [1] Born Rule
    # =================================================================
    print("=" * 65)
    print("  [1] Born Rule: |psi_i|^2 -> probabilities")
    print("=" * 65)
    print(f"  {'dim':>6s} | {'Quantum Sim':>12s} | {'AXOL':>12s} | {'Ratio':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for dim in dims_large:
        psi = make_state(dim)
        q_us = bench_quantum_sim("born", born_rule, ITERATIONS, psi)
        a_us = axol.get("Born rule (measure)", {}).get(dim, None)

        if a_us is not None:
            ratio = q_us / a_us
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {a_us:>10.3f}us | {ratio:>7.1f}x")
        else:
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {'N/A':>12s} | {'':>8s}")

    # =================================================================
    # [2] Density Matrix + Purity
    # =================================================================
    print()
    print("=" * 65)
    print("  [2] Density Matrix (rho = |psi><psi|) + Purity Tr(rho^2)")
    print("=" * 65)
    print(f"  {'dim':>6s} | {'Quantum Sim':>12s} | {'AXOL':>12s} | {'Ratio':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for dim in dims_small:
        psi = make_state(dim)
        def density_and_purity(p):
            rho = density_matrix(p)
            return purity(rho)

        iters = ITERATIONS if dim <= 16 else 50_000
        q_us = bench_quantum_sim("density", density_and_purity, iters, psi)
        a_us = axol.get("Density matrix (from_pure + purity)", {}).get(dim, None)

        if a_us is not None:
            ratio = q_us / a_us
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {a_us:>10.3f}us | {ratio:>7.1f}x")
        else:
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {'N/A':>12s} | {'':>8s}")

    # =================================================================
    # [3] Interference
    # =================================================================
    print()
    print("=" * 65)
    print("  [3] Interference: alpha*|psi1> + beta*|psi2>")
    print("=" * 65)
    print(f"  {'dim':>6s} | {'Quantum Sim':>12s} | {'AXOL':>12s} | {'Ratio':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for dim in dims_large:
        psi1 = make_state(dim)
        psi2 = make_state(dim)[::-1]  # reversed
        psi2 /= np.linalg.norm(psi2)

        q_us = bench_quantum_sim("interfere", interfere, ITERATIONS, psi1, psi2, 0.3)
        a_us = axol.get("Interference", {}).get(dim, None)

        if a_us is not None:
            ratio = q_us / a_us
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {a_us:>10.3f}us | {ratio:>7.1f}x")
        else:
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {'N/A':>12s} | {'':>8s}")

    # =================================================================
    # [4] Dephasing Channel
    # =================================================================
    print()
    print("=" * 65)
    print("  [4] Dephasing Channel (Kraus operators)")
    print("=" * 65)
    print(f"  {'dim':>6s} | {'Quantum Sim':>12s} | {'AXOL':>12s} | {'Ratio':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for dim in dims_small:
        psi = make_state(dim)
        rho = density_matrix(psi)
        kraus = dephasing_kraus(0.3, dim)

        iters = 10_000
        q_us = bench_quantum_sim("dephase", apply_channel, iters, rho, kraus)
        a_us = axol.get("Dephasing channel (apply)", {}).get(dim, None)

        if a_us is not None:
            ratio = q_us / a_us
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {a_us:>10.3f}us | {ratio:>7.1f}x")
        else:
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {'N/A':>12s} | {'':>8s}")

    # =================================================================
    # [5] Von Neumann Entropy
    # =================================================================
    print()
    print("=" * 65)
    print("  [5] Von Neumann Entropy: -Tr(rho * log(rho))")
    print("=" * 65)
    print(f"  {'dim':>6s} | {'Quantum Sim':>12s} | {'AXOL':>12s} | {'Ratio':>8s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for dim in [4, 8, 16]:
        psi = make_state(dim)
        rho = density_matrix(psi)

        iters = 10_000
        q_us = bench_quantum_sim("entropy", von_neumann_entropy, iters, rho)
        a_us = axol.get("Von Neumann entropy", {}).get(dim, None)

        if a_us is not None:
            ratio = q_us / a_us
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {a_us:>10.3f}us | {ratio:>7.1f}x")
        else:
            print(f"  {dim:>6d} | {q_us:>10.3f}us | {'N/A':>12s} | {'':>8s}")

    # =================================================================
    # Summary
    # =================================================================
    print()
    print("#" * 65)
    print("#  SUMMARY")
    print("#" * 65)
    print()
    print("  Quantum Sim = numpy statevector simulator (Python)")
    print("  AXOL        = Rust-native chaos engine")
    print("  Ratio       = Quantum Sim time / AXOL time (higher = AXOL faster)")
    print()
    print("  Both run the SAME quantum math on CLASSICAL hardware.")
    print("  AXOL is purpose-built for these operations in Rust.")
    print("  Quantum simulators are general-purpose Python wrappers.")
    print()
    print("  Real quantum hardware (IBM, Google) would:")
    print("    - Be faster for large dim (exponential advantage)")
    print("    - But require cryogenic cooling, error correction")
    print("    - And cost $$$$ per shot")
    print()
    print("  AXOL sits between classical simulators and quantum hardware:")
    print("    Classical Sim  <  AXOL (Rust-native)  <  Quantum Hardware")
    print("    (slow, general)   (fast, specialized)    (fastest, expensive)")
    print()
    print("#" * 65)


if __name__ == "__main__":
    main()
