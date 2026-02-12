"""Benchmark: AXOL Quantum vs Pure Python vs NumPy(C) — SVG output."""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from axol.quantum import *
from axol.core.types import FloatVec, ComplexVec, DensityMatrix, TransMatrix
from axol.core import operations as ops
from axol.quantum.density import (
    von_neumann_entropy, fidelity, apply_channel,
    depolarizing_channel, amplitude_damping_channel, dephasing_channel,
    phi_from_purity, omega_from_coherence,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'benchmark_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ──
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})
COLORS = {
    'python': '#3572A5',
    'numpy': '#4B8BBE',
    'axol': '#f0883e',
    'axol_q': '#da3633',
}


def timeit(fn, N=100, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    return (time.perf_counter() - t0) / N * 1e6  # microseconds


# =====================================================================
# Benchmark 1: Vector-scalar operations
# =====================================================================
def bench_vector_ops():
    dims = [4, 8, 16, 32, 64, 128, 256]
    results = {k: [] for k in ['python', 'numpy', 'axol', 'axol_q']}

    for d in dims:
        data = np.random.randn(d).astype(np.float32)
        data_c = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        py_list = data.tolist()
        fv = FloatVec(data=data)
        cv = ComplexVec(data=data_c / np.linalg.norm(data_c))

        # Pure Python: normalize list
        def py_fn():
            s = sum(x * x for x in py_list)
            s = s ** 0.5
            return [x / s for x in py_list]

        # NumPy (C): normalize array
        def np_fn():
            return data / np.linalg.norm(data)

        # AXOL: measure (Born rule on real)
        def axol_fn():
            return ops.measure(fv)

        # AXOL quantum: measure_complex (Born rule on complex)
        def axol_q_fn():
            return ops.measure_complex(cv)

        results['python'].append(timeit(py_fn, N=200))
        results['numpy'].append(timeit(np_fn, N=200))
        results['axol'].append(timeit(axol_fn, N=200))
        results['axol_q'].append(timeit(axol_q_fn, N=200))

    return dims, results, 'Born Rule / Normalize', 'bench_1_vector_ops.svg'


# =====================================================================
# Benchmark 2: Matrix-vector multiply
# =====================================================================
def bench_matmul():
    dims = [4, 8, 16, 32, 64, 128]
    results = {k: [] for k in ['python', 'numpy', 'axol', 'axol_q']}

    for d in dims:
        mat = np.random.randn(d, d).astype(np.float32)
        vec = np.random.randn(d).astype(np.float32)
        py_mat = mat.tolist()
        py_vec = vec.tolist()
        tm = TransMatrix(data=mat)
        fv = FloatVec(data=vec)
        cv = ComplexVec(data=vec.astype(np.complex128))

        # Pure Python
        def py_fn():
            result = []
            for i in range(d):
                s = 0.0
                for j in range(d):
                    s += py_vec[j] * py_mat[i][j]
                result.append(s)
            return result

        # NumPy (C)
        def np_fn():
            return vec @ mat

        # AXOL: transform
        def axol_fn():
            return ops.transform(fv, tm)

        # AXOL quantum: transform_complex
        def axol_q_fn():
            return ops.transform_complex(cv, tm)

        N = 200 if d <= 64 else 50
        results['python'].append(timeit(py_fn, N=N))
        results['numpy'].append(timeit(np_fn, N=N))
        results['axol'].append(timeit(axol_fn, N=N))
        results['axol_q'].append(timeit(axol_q_fn, N=N))

    return dims, results, 'Matrix-Vector Multiply', 'bench_2_matmul.svg'


# =====================================================================
# Benchmark 3: Density matrix operations (quantum-specific)
# =====================================================================
def bench_density():
    dims = [4, 8, 16, 32, 64]
    results = {k: [] for k in ['python', 'numpy', 'axol_q']}

    for d in dims:
        np.random.seed(42)
        psi_data = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        psi_data /= np.linalg.norm(psi_data)
        cv = ComplexVec(data=psi_data)

        # Pure Python: outer product + trace(rho^2)
        def py_fn():
            psi = psi_data.tolist()
            dim = len(psi)
            rho = [[psi[i] * psi[j].conjugate() for j in range(dim)] for i in range(dim)]
            purity = 0
            for i in range(dim):
                for j in range(dim):
                    purity += (rho[i][j] * rho[j][i]).real
            return purity

        # NumPy (C): outer product + trace
        def np_fn():
            rho = np.outer(psi_data, psi_data.conj())
            return np.real(np.trace(rho @ rho))

        # AXOL quantum: DensityMatrix + purity
        def axol_q_fn():
            rho = DensityMatrix.from_pure_state(cv)
            return rho.purity

        N = 100 if d <= 32 else 30
        results['python'].append(timeit(py_fn, N=N))
        results['numpy'].append(timeit(np_fn, N=N))
        results['axol_q'].append(timeit(axol_q_fn, N=N))

    return dims, results, 'Density Matrix (pure state + purity)', 'bench_3_density.svg'


# =====================================================================
# Benchmark 4: Quantum channel application
# =====================================================================
def bench_channels():
    dims = [4, 8, 16, 32]
    results = {k: [] for k in ['numpy', 'axol_q']}

    for d in dims:
        np.random.seed(42)
        psi_data = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        psi_data /= np.linalg.norm(psi_data)
        rho_data = np.outer(psi_data, psi_data.conj())
        rho = DensityMatrix(data=rho_data)
        kraus = dephasing_channel(0.3, d)
        kraus_np = [k.copy() for k in kraus]

        # NumPy (C): manual Kraus apply
        def np_fn():
            result = np.zeros_like(rho_data)
            for E in kraus_np:
                result += E @ rho_data @ E.conj().T
            return result

        # AXOL quantum: apply_channel
        def axol_q_fn():
            return apply_channel(rho, kraus)

        N = 100 if d <= 16 else 30
        results['numpy'].append(timeit(np_fn, N=N))
        results['axol_q'].append(timeit(axol_q_fn, N=N))

    return dims, results, 'Quantum Channel (dephasing)', 'bench_4_channels.svg'


# =====================================================================
# Benchmark 5: Interference
# =====================================================================
def bench_interference():
    dims = [4, 16, 64, 128, 256, 512]
    results = {k: [] for k in ['python', 'numpy', 'axol_q']}

    for d in dims:
        np.random.seed(0)
        a_data = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        b_data = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        a_data /= np.linalg.norm(a_data)
        b_data /= np.linalg.norm(b_data)
        a_list = a_data.tolist()
        b_list = b_data.tolist()
        phase = 0.3
        exp_phase = np.exp(1j * phase)

        a_cv = ComplexVec(data=a_data)
        b_cv = ComplexVec(data=b_data)

        # Pure Python
        def py_fn():
            import cmath
            ep = cmath.exp(1j * phase)
            r = [a_list[i] + ep * b_list[i] for i in range(d)]
            norm = sum(abs(x) ** 2 for x in r) ** 0.5
            return [x / norm for x in r]

        # NumPy (C)
        def np_fn():
            r = a_data + exp_phase * b_data
            return r / np.linalg.norm(r)

        # AXOL quantum
        def axol_q_fn():
            return ops.interfere(a_cv, b_cv, phase=phase)

        results['python'].append(timeit(py_fn, N=200))
        results['numpy'].append(timeit(np_fn, N=200))
        results['axol_q'].append(timeit(axol_q_fn, N=200))

    return dims, results, 'Quantum Interference', 'bench_5_interference.svg'


# =====================================================================
# Benchmark 6: End-to-end pipeline (AXOL-only, classical vs quantum)
# =====================================================================
def bench_pipeline():
    dims = [4, 8, 16]

    weave_c, weave_q = [], []
    observe_c, observe_q = [], []
    reobserve_c, reobserve_q = [], []

    for d in dims:
        b = DeclarationBuilder(f'pipe{d}')
        b.input('x', d)
        b.input('y', d)
        b.relate('z', ['x', 'y'], RelationKind.PROPORTIONAL)
        b.output('z')
        b.quality(omega=0.8, phi=0.8)
        decl = b.build()

        inp = {'x': FloatVec.from_list([1.0] * d), 'y': FloatVec.from_list([0.5] * d)}

        # Weave
        t0 = time.perf_counter()
        tap_c = weave(decl, quantum=False, seed=42)
        weave_c.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        tap_q = weave(decl, quantum=True, seed=42)
        weave_q.append((time.perf_counter() - t0) * 1000)

        # Observe (average 10 runs)
        t0 = time.perf_counter()
        for _ in range(10):
            observe(tap_c, inp)
        observe_c.append((time.perf_counter() - t0) / 10 * 1000)

        t0 = time.perf_counter()
        for _ in range(10):
            observe(tap_q, inp)
        observe_q.append((time.perf_counter() - t0) / 10 * 1000)

        # Reobserve (count=5)
        t0 = time.perf_counter()
        reobserve(tap_c, inp, count=5)
        reobserve_c.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        reobserve(tap_q, inp, count=5)
        reobserve_q.append((time.perf_counter() - t0) * 1000)

    return dims, weave_c, weave_q, observe_c, observe_q, reobserve_c, reobserve_q


# =====================================================================
# Plotting helpers
# =====================================================================

def plot_comparison(dims, results, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(dims))
    width = 0.2
    offsets = {'python': -1.5, 'numpy': -0.5, 'axol': 0.5, 'axol_q': 1.5}
    labels = {'python': 'Pure Python', 'numpy': 'NumPy (C)', 'axol': 'AXOL Classical', 'axol_q': 'AXOL Quantum'}

    for key in results:
        offset = offsets.get(key, 0) * width
        bars = ax.bar(x + offset, results[key], width * 0.9,
                      label=labels.get(key, key), color=COLORS.get(key, '#888'),
                      edgecolor='none', alpha=0.9)
        # value labels on bars
        for bar, val in zip(bars, results[key]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.0f}' if val >= 10 else f'{val:.1f}',
                        ha='center', va='bottom', fontsize=7, color='#8b949e')

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Latency (us)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(OUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, format='svg', dpi=150)
    plt.close(fig)
    print(f'  -> {path}')
    return path


def plot_pipeline(dims, wc, wq, oc, oq, rc, rq):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(dims))
    w = 0.35

    titles = ['Weave', 'Observe', 'Reobserve (count=5)']
    data_pairs = [(wc, wq), (oc, oq), (rc, rq)]

    for ax, (dc, dq), title in zip(axes, data_pairs, titles):
        bars1 = ax.bar(x - w / 2, dc, w, label='Classical', color=COLORS['axol'], alpha=0.9, edgecolor='none')
        bars2 = ax.bar(x + w / 2, dq, w, label='Quantum', color=COLORS['axol_q'], alpha=0.9, edgecolor='none')

        for bar, val in zip(list(bars1) + list(bars2), list(dc) + list(dq)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#8b949e')

        ax.set_xlabel('Dimension')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    path = os.path.join(OUT_DIR, 'bench_6_pipeline.svg')
    fig.suptitle('AXOL Pipeline: Classical vs Quantum', fontsize=14, fontweight='bold', color='#c9d1d9')
    fig.tight_layout()
    fig.savefig(path, format='svg', dpi=150)
    plt.close(fig)
    print(f'  -> {path}')
    return path


# =====================================================================
# Quality metrics summary chart
# =====================================================================
def plot_quality():
    dims_q = [4, 8, 16]
    omegas_c, phis_c = [], []
    omegas_q, phis_q = [], []
    q_phis, q_omegas = [], []
    purities, entropies = [], []

    for d in dims_q:
        b = DeclarationBuilder(f'q{d}')
        b.input('x', d)
        b.input('y', d)
        b.relate('z', ['x', 'y'], RelationKind.PROPORTIONAL)
        b.output('z')
        b.quality(omega=0.8, phi=0.8)
        decl = b.build()
        inp = {'x': FloatVec.from_list([1.0] * d), 'y': FloatVec.from_list([0.5] * d)}

        tap_c = weave(decl, quantum=False, seed=42)
        tap_q = weave(decl, quantum=True, seed=42)
        obs_c = observe(tap_c, inp)
        obs_q = observe(tap_q, inp)

        omegas_c.append(obs_c.omega)
        phis_c.append(obs_c.phi)
        omegas_q.append(obs_q.omega)
        phis_q.append(obs_q.phi)
        q_phis.append(obs_q.quantum_phi if obs_q.quantum_phi is not None else 0)
        q_omegas.append(obs_q.quantum_omega if obs_q.quantum_omega is not None else 0)
        if obs_q.density_matrix is not None:
            purities.append(obs_q.density_matrix.purity)
            entropies.append(von_neumann_entropy(obs_q.density_matrix))
        else:
            purities.append(0)
            entropies.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Omega/Phi classical vs quantum
    x = np.arange(len(dims_q))
    w = 0.2
    ax = axes[0]
    ax.bar(x - 1.5 * w, omegas_c, w, label='Omega (Classical)', color='#58a6ff', alpha=0.9, edgecolor='none')
    ax.bar(x - 0.5 * w, omegas_q, w, label='Omega (Quantum)', color='#1f6feb', alpha=0.9, edgecolor='none')
    ax.bar(x + 0.5 * w, phis_c, w, label='Phi (Classical)', color='#f0883e', alpha=0.9, edgecolor='none')
    ax.bar(x + 1.5 * w, phis_q, w, label='Phi (Quantum)', color='#da3633', alpha=0.9, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims_q])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Omega & Phi')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Quantum-only metrics
    ax = axes[1]
    ax.bar(x - w / 2, q_omegas, w, label='Q-Omega (coherence)', color='#1f6feb', alpha=0.9, edgecolor='none')
    ax.bar(x + w / 2, q_phis, w, label='Q-Phi (purity)', color='#da3633', alpha=0.9, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims_q])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Quantum Quality Metrics')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Purity & Entropy
    ax = axes[2]
    ax2 = ax.twinx()
    b1 = ax.bar(x - w / 2, purities, w, label='Purity', color='#3fb950', alpha=0.9, edgecolor='none')
    b2 = ax2.bar(x + w / 2, entropies, w, label='vN Entropy', color='#bc8cff', alpha=0.9, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims_q])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Purity', color='#3fb950')
    ax2.set_ylabel('Entropy', color='#bc8cff')
    ax.set_title('Density Matrix Properties')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(OUT_DIR, 'bench_7_quality.svg')
    fig.suptitle('AXOL Quality Metrics: Classical vs Quantum', fontsize=14, fontweight='bold', color='#c9d1d9')
    fig.tight_layout()
    fig.savefig(path, format='svg', dpi=150)
    plt.close(fig)
    print(f'  -> {path}')
    return path


# =====================================================================
# Main
# =====================================================================
def main():
    print('=' * 60)
    print('  AXOL Quantum Benchmark — Python vs NumPy(C) vs AXOL')
    print('=' * 60)

    print('\n[1/7] Vector / Born rule...')
    d, r, t, f = bench_vector_ops()
    plot_comparison(d, r, t, f)

    print('[2/7] Matrix-vector multiply...')
    d, r, t, f = bench_matmul()
    plot_comparison(d, r, t, f)

    print('[3/7] Density matrix...')
    d, r, t, f = bench_density()
    plot_comparison(d, r, t, f)

    print('[4/7] Quantum channel...')
    d, r, t, f = bench_channels()
    plot_comparison(d, r, t, f)

    print('[5/7] Interference...')
    d, r, t, f = bench_interference()
    plot_comparison(d, r, t, f)

    print('[6/7] Pipeline (classical vs quantum)...')
    dims, wc, wq, oc, oq, rc, rq = bench_pipeline()
    plot_pipeline(dims, wc, wq, oc, oq, rc, rq)

    print('[7/7] Quality metrics...')
    plot_quality()

    print(f'\nAll SVGs saved to: {os.path.abspath(OUT_DIR)}')
    print('=' * 60)


if __name__ == '__main__':
    main()
