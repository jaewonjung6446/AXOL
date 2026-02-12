"""Lightweight benchmark for Direction B quantum structures."""
import time, sys, numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from axol.quantum import *
from axol.core.types import FloatVec, ComplexVec, DensityMatrix, TransMatrix
from axol.core import operations as ops
from axol.quantum.density import (
    von_neumann_entropy, fidelity, apply_channel,
    depolarizing_channel, amplitude_damping_channel, dephasing_channel,
    svd_to_kraus, phi_from_purity, omega_from_coherence,
)


def build_decl(dim=8, name='bench'):
    b = DeclarationBuilder(name)
    b.input('x', dim)
    b.input('y', dim)
    b.relate('z', ['x', 'y'], RelationKind.PROPORTIONAL)
    b.output('z')
    b.quality(omega=0.8, phi=0.8)
    return b.build()


def main():
    print('=' * 70)
    print('  AXOL Quantum Structures (Direction B) - Lightweight Benchmark')
    print('=' * 70)

    # =================================================================
    # 1. Weave: classical vs quantum (1 run each, small dims)
    # =================================================================
    print('\n[1] Weave latency (single run)')
    print(f'  {"dim":>3} | {"classical(ms)":>13} | {"quantum(ms)":>11} | {"overhead":>8}')
    print('  ' + '-' * 46)

    for d in [4, 8, 16]:
        decl = build_decl(d, f'w{d}')
        t0 = time.perf_counter()
        tap_c = weave(decl, quantum=False, seed=42)
        t_c = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tap_q = weave(decl, quantum=True, seed=42)
        t_q = (time.perf_counter() - t0) * 1000

        ratio = t_q / t_c if t_c > 0 else float('inf')
        print(f'  {d:>3} | {t_c:>13.2f} | {t_q:>11.2f} | {ratio:>7.2f}x')

    # =================================================================
    # 2. Observe: classical vs quantum
    # =================================================================
    print('\n[2] Observe latency (20 iterations)')
    print(f'  {"dim":>3} | {"classical(us)":>13} | {"quantum(us)":>11} | {"overhead":>8}')
    print('  ' + '-' * 46)

    for d in [4, 8, 16]:
        decl = build_decl(d, f'o{d}')
        tap_c = weave(decl, quantum=False, seed=42)
        tap_q = weave(decl, quantum=True, seed=42)
        inp = {'x': FloatVec.from_list([1.0] * d), 'y': FloatVec.from_list([0.5] * d)}

        t0 = time.perf_counter()
        for _ in range(20):
            obs_c = observe(tap_c, inp)
        t_c = (time.perf_counter() - t0) / 20 * 1e6

        t0 = time.perf_counter()
        for _ in range(20):
            obs_q = observe(tap_q, inp)
        t_q = (time.perf_counter() - t0) / 20 * 1e6

        ratio = t_q / t_c if t_c > 0 else float('inf')
        print(f'  {d:>3} | {t_c:>13.1f} | {t_q:>11.1f} | {ratio:>7.2f}x')

    # =================================================================
    # 3. Quality metrics comparison
    # =================================================================
    print('\n[3] Quality metrics (dim=8)')
    decl = build_decl(8, 'qm')
    tap_c = weave(decl, quantum=False, seed=42)
    tap_q = weave(decl, quantum=True, seed=42)
    inp = {'x': FloatVec.from_list([1.0] * 8), 'y': FloatVec.from_list([0.5] * 8)}

    obs_c = observe(tap_c, inp)
    obs_q = observe(tap_q, inp)

    print(f'  Classical:  omega={obs_c.omega:.4f}  phi={obs_c.phi:.4f}  idx={obs_c.value_index}')
    print(f'  Quantum:    omega={obs_q.omega:.4f}  phi={obs_q.phi:.4f}  idx={obs_q.value_index}')
    if obs_q.quantum_phi is not None:
        print(f'  Q-metrics:  q_omega={obs_q.quantum_omega:.4f}  q_phi={obs_q.quantum_phi:.4f}')
    if obs_q.density_matrix is not None:
        rho = obs_q.density_matrix
        print(f'  Density:    purity={rho.purity:.4f}  vN_entropy={von_neumann_entropy(rho):.4f}')

    # =================================================================
    # 4. Density matrix ops (pure numpy, no weave)
    # =================================================================
    print('\n[4] Density matrix ops (100 iterations)')
    print(f'  {"dim":>3} | {"pure_state(us)":>14} | {"entropy(us)":>11} | {"fidelity(us)":>12} | {"channel(us)":>11}')
    print('  ' + '-' * 60)

    for d in [4, 8, 16, 32]:
        np.random.seed(0)
        psi = ComplexVec(data=(np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128))
        psi = ComplexVec(data=psi.data / np.linalg.norm(psi.data))

        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            rho = DensityMatrix.from_pure_state(psi)
        t_pure = (time.perf_counter() - t0) / N * 1e6

        t0 = time.perf_counter()
        for _ in range(N):
            von_neumann_entropy(rho)
        t_ent = (time.perf_counter() - t0) / N * 1e6

        rho2 = DensityMatrix.from_pure_state(ComplexVec(data=psi.data * np.exp(1j * 0.1)))
        t0 = time.perf_counter()
        for _ in range(N):
            fidelity(rho, rho2)
        t_fid = (time.perf_counter() - t0) / N * 1e6

        kraus = depolarizing_channel(d, 0.1)
        t0 = time.perf_counter()
        for _ in range(N):
            apply_channel(rho, kraus)
        t_ch = (time.perf_counter() - t0) / N * 1e6

        print(f'  {d:>3} | {t_pure:>14.1f} | {t_ent:>11.1f} | {t_fid:>12.1f} | {t_ch:>11.1f}')

    # =================================================================
    # 5. Interference
    # =================================================================
    print('\n[5] Interference (200 iterations)')
    print(f'  {"dim":>3} | {"interfere(us)":>13} | {"measure(us)":>11}')
    print('  ' + '-' * 34)

    for d in [4, 16, 64, 256]:
        np.random.seed(1)
        a = ComplexVec(data=(np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128))
        b = ComplexVec(data=(np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128))
        a = ComplexVec(data=a.data / np.linalg.norm(a.data))
        b = ComplexVec(data=b.data / np.linalg.norm(b.data))

        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            c = ops.interfere(a, b, phase=0.3)
        t_int = (time.perf_counter() - t0) / N * 1e6

        t0 = time.perf_counter()
        for _ in range(N):
            ops.measure_complex(c)
        t_meas = (time.perf_counter() - t0) / N * 1e6

        print(f'  {d:>3} | {t_int:>13.1f} | {t_meas:>11.1f}')

    # =================================================================
    # 6. Quantum channels
    # =================================================================
    print('\n[6] Quantum channels (dim=8, 100 iterations)')
    d = 8
    np.random.seed(2)
    psi = ComplexVec(data=(np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128))
    psi = ComplexVec(data=psi.data / np.linalg.norm(psi.data))
    rho = DensityMatrix.from_pure_state(psi)

    channels = {
        'depolarizing': depolarizing_channel(d, 0.3),
        'amp_damping': amplitude_damping_channel(0.3, d),
        'dephasing': dephasing_channel(0.3, d),
    }
    for name, kraus in channels.items():
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            r = apply_channel(rho, kraus)
        t = (time.perf_counter() - t0) / N * 1e6
        print(f'  {name:<14} | {len(kraus):>3} Kraus ops | {t:>8.1f} us | purity={r.purity:.4f}')

    # =================================================================
    # 7. Reobserve
    # =================================================================
    print('\n[7] Reobserve (count=5, dim=8)')
    decl = build_decl(8, 'ro')
    tap_c = weave(decl, quantum=False, seed=42)
    tap_q = weave(decl, quantum=True, seed=42)
    inp = {'x': FloatVec.from_list([1.0] * 8), 'y': FloatVec.from_list([0.5] * 8)}

    t0 = time.perf_counter()
    robs_c = reobserve(tap_c, inp, count=5)
    t_c = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    robs_q = reobserve(tap_q, inp, count=5)
    t_q = (time.perf_counter() - t0) * 1000

    print(f'  Classical: {t_c:.1f} ms  omega={robs_c.omega:.4f}  phi={robs_c.phi:.4f}')
    print(f'  Quantum:   {t_q:.1f} ms  omega={robs_q.omega:.4f}  phi={robs_q.phi:.4f}')

    print('\n' + '=' * 70)
    print('  Benchmark complete.')
    print('=' * 70)


if __name__ == '__main__':
    main()
