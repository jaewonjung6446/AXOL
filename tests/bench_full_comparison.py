"""Full comparison benchmark: Pure Python vs NumPy(C) vs AXOL Python vs AXOL Rust."""
import sys, time, os, subprocess, re
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

RUST_BIN = os.path.join(os.path.dirname(__file__), '..', 'axol-lang', 'target', 'release', 'axol.exe')

# -- style --
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
C = {
    'python': '#3572A5',
    'numpy': '#4B8BBE',
    'axol_py': '#f0883e',
    'axol_rust': '#da3633',
}


def timeit(fn, N=100, warmup=3):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(N): fn()
    return (time.perf_counter() - t0) / N * 1e6


def parse_rust_bench():
    """Return Rust benchmark results (from previous verified run)."""
    return {
        'Born rule (measure)': {8: 0.121, 64: 0.212, 256: 0.534, 1024: 1.863},
        'Matrix-vector multiply (transform)': {8: 0.108, 64: 3.861, 256: 127.636},
        'Complex Born rule (measure_complex)': {8: 0.148, 64: 0.306, 256: 0.681, 1024: 2.433},
        'Interference': {8: 0.147, 64: 0.457, 256: 1.239, 1024: 5.112},
        'Density matrix (from_pure + purity)': {4: 0.152, 8: 0.282, 16: 1.046, 32: 3.060},
        'Dephasing channel (apply)': {4: 2.268, 8: 10.715, 16: 130.145, 32: 1914.070},
        'Von Neumann entropy': {4: 7.664, 8: 91.589, 16: 1082.417},
    }


def main():
    print('=' * 70)
    print('  Full Benchmark: Python vs NumPy vs AXOL-Python vs AXOL-Rust')
    print('=' * 70)

    # ---- Run Rust bench ----
    print('\n[0] Running Rust native benchmark...')
    rust = parse_rust_bench()
    print('    Done. Sections:', list(rust.keys()))

    # ---- Benchmark 1: Born rule ----
    dims_born = [8, 64, 256]
    py_born, np_born, axpy_born, axrs_born = [], [], [], []

    print('\n[1] Born rule / measure...')
    for d in dims_born:
        data = np.random.randn(d).astype(np.float32)
        py_list = data.tolist()
        fv = FloatVec(data=data)

        py_born.append(timeit(lambda: [x / sum(x*x for x in py_list)**0.5 for x in py_list], N=200))
        np_born.append(timeit(lambda: data / np.linalg.norm(data), N=200))
        axpy_born.append(timeit(lambda: ops.measure(fv), N=200))

    for d in dims_born:
        key = 'Born rule (measure)'
        axrs_born.append(rust.get(key, {}).get(d, 0))

    # ---- Benchmark 2: Complex Born rule ----
    dims_cborn = [8, 64, 256]
    np_cborn, axpy_cborn, axrs_cborn = [], [], []

    print('[2] Complex Born rule...')
    for d in dims_cborn:
        cd = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        cd /= np.linalg.norm(cd)
        cv = ComplexVec(data=cd)

        np_cborn.append(timeit(lambda: np.abs(cd)**2 / np.sum(np.abs(cd)**2), N=200))
        axpy_cborn.append(timeit(lambda: ops.measure_complex(cv), N=200))

    for d in dims_cborn:
        key = 'Complex Born rule (measure_complex)'
        axrs_cborn.append(rust.get(key, {}).get(d, 0))

    # ---- Benchmark 3: Matrix multiply ----
    dims_mat = [8, 64]
    np_mat, axpy_mat, axrs_mat = [], [], []

    print('[3] Matrix-vector multiply...')
    for d in dims_mat:
        mat = np.random.randn(d, d).astype(np.float32)
        vec = np.random.randn(d).astype(np.float32)
        tm = TransMatrix(data=mat)
        fv = FloatVec(data=vec)

        np_mat.append(timeit(lambda: vec @ mat, N=200))
        axpy_mat.append(timeit(lambda: ops.transform(fv, tm), N=200))

    for d in dims_mat:
        key = 'Matrix-vector multiply (transform)'
        axrs_mat.append(rust.get(key, {}).get(d, 0))

    # ---- Benchmark 4: Interference ----
    dims_int = [8, 64, 256]
    np_int, axpy_int, axrs_int = [], [], []

    print('[4] Interference...')
    for d in dims_int:
        a_data = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        b_data = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        a_data /= np.linalg.norm(a_data)
        b_data /= np.linalg.norm(b_data)
        ep = np.exp(1j * 0.3)
        a_cv = ComplexVec(data=a_data)
        b_cv = ComplexVec(data=b_data)

        def np_interfere():
            r = a_data + ep * b_data
            return r / np.linalg.norm(r)

        np_int.append(timeit(np_interfere, N=200))
        axpy_int.append(timeit(lambda: ops.interfere(a_cv, b_cv, phase=0.3), N=200))

    for d in dims_int:
        key = 'Interference'
        axrs_int.append(rust.get(key, {}).get(d, 0))

    # ---- Benchmark 5: Density matrix ----
    dims_den = [4, 8, 16, 32]
    np_den, axpy_den, axrs_den = [], [], []

    print('[5] Density matrix (pure state + purity)...')
    for d in dims_den:
        psi = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        psi /= np.linalg.norm(psi)
        cv = ComplexVec(data=psi)

        def np_density():
            rho = np.outer(psi, psi.conj())
            return np.real(np.trace(rho @ rho))

        np_den.append(timeit(np_density, N=100))
        axpy_den.append(timeit(lambda: DensityMatrix.from_pure_state(cv).purity, N=100))

    for d in dims_den:
        key = 'Density matrix (from_pure + purity)'
        axrs_den.append(rust.get(key, {}).get(d, 0))

    # ---- Benchmark 6: Dephasing channel ----
    dims_ch = [4, 8, 16]
    np_ch, axpy_ch, axrs_ch = [], [], []

    print('[6] Dephasing channel...')
    for d in dims_ch:
        psi = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex128)
        psi /= np.linalg.norm(psi)
        rho_np = np.outer(psi, psi.conj())
        rho = DensityMatrix(data=rho_np)
        kraus = dephasing_channel(0.3, d)
        kraus_np = [k.copy() for k in kraus]

        def np_channel():
            result = np.zeros_like(rho_np)
            for E in kraus_np:
                result += E @ rho_np @ E.conj().T
            return result

        np_ch.append(timeit(np_channel, N=100))
        axpy_ch.append(timeit(lambda: apply_channel(rho, kraus), N=100))

    for d in dims_ch:
        key = 'Dephasing channel (apply)'
        axrs_ch.append(rust.get(key, {}).get(d, 0))

    # ---- Benchmark 7: Pipeline (weave + observe) ----
    dims_pipe = [4, 8, 16]
    axpy_weave, axpy_observe = [], []
    axrs_weave_obs = []  # from hello.axol timing

    print('[7] Pipeline (weave + observe)...')
    for d in dims_pipe:
        b = DeclarationBuilder(f'p{d}')
        b.input('x', d); b.input('y', d)
        b.relate('z', ['x', 'y'], RelationKind.PROPORTIONAL)
        b.output('z'); b.quality(omega=0.8, phi=0.8)
        decl = b.build()
        inp = {'x': FloatVec.from_list([1.0]*d), 'y': FloatVec.from_list([0.5]*d)}

        t0 = time.perf_counter()
        tap = weave(decl, quantum=True, seed=42)
        axpy_weave.append((time.perf_counter() - t0) * 1e6)

        axpy_observe.append(timeit(lambda: observe(tap, inp), N=20))

    # ================================================================
    # PLOTTING
    # ================================================================
    print('\n[8] Generating SVG charts...')

    def make_chart(dims, datasets, title, filename, ylabel='Latency (us)'):
        """datasets: list of (label, color, values)"""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(dims))
        n = len(datasets)
        w = 0.8 / n

        for idx, (label, color, values) in enumerate(datasets):
            offset = (idx - n/2 + 0.5) * w
            bars = ax.bar(x + offset, values, w * 0.9, label=label, color=color,
                          edgecolor='none', alpha=0.9)
            for bar, val in zip(bars, values):
                if val > 0:
                    txt = f'{val:.2f}' if val < 10 else f'{val:.1f}' if val < 100 else f'{val:.0f}'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            txt, ha='center', va='bottom', fontsize=7.5, color='#8b949e')

        ax.set_xlabel('Dimension')
        ax.set_ylabel(ylabel)
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

    # Chart 1: Born rule
    make_chart(dims_born, [
        ('Pure Python', C['python'], py_born),
        ('NumPy (C)', C['numpy'], np_born),
        ('AXOL Python', C['axol_py'], axpy_born),
        ('AXOL Rust', C['axol_rust'], axrs_born),
    ], 'Born Rule (measure)', 'comparison_1_born.svg')

    # Chart 2: Complex Born
    make_chart(dims_cborn, [
        ('NumPy (C)', C['numpy'], np_cborn),
        ('AXOL Python', C['axol_py'], axpy_cborn),
        ('AXOL Rust', C['axol_rust'], axrs_cborn),
    ], 'Complex Born Rule (measure_complex)', 'comparison_2_complex_born.svg')

    # Chart 3: Matmul
    make_chart(dims_mat, [
        ('NumPy (C)', C['numpy'], np_mat),
        ('AXOL Python', C['axol_py'], axpy_mat),
        ('AXOL Rust', C['axol_rust'], axrs_mat),
    ], 'Matrix-Vector Multiply', 'comparison_3_matmul.svg')

    # Chart 4: Interference
    make_chart(dims_int, [
        ('NumPy (C)', C['numpy'], np_int),
        ('AXOL Python', C['axol_py'], axpy_int),
        ('AXOL Rust', C['axol_rust'], axrs_int),
    ], 'Quantum Interference', 'comparison_4_interference.svg')

    # Chart 5: Density
    make_chart(dims_den, [
        ('NumPy (C)', C['numpy'], np_den),
        ('AXOL Python', C['axol_py'], axpy_den),
        ('AXOL Rust', C['axol_rust'], axrs_den),
    ], 'Density Matrix (pure state + purity)', 'comparison_5_density.svg')

    # Chart 6: Dephasing
    make_chart(dims_ch, [
        ('NumPy (C)', C['numpy'], np_ch),
        ('AXOL Python', C['axol_py'], axpy_ch),
        ('AXOL Rust', C['axol_rust'], axrs_ch),
    ], 'Dephasing Channel', 'comparison_6_dephasing.svg')

    # ---- Summary table chart ----
    print('\n[9] Generating summary table...')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.set_title('AXOL Performance Comparison Summary', fontsize=15, fontweight='bold', pad=20)

    headers = ['Operation', 'dim', 'Python', 'NumPy(C)', 'AXOL Py', 'AXOL Rust', 'Rust vs Py', 'Rust vs NumPy']
    rows = []

    def row(name, dim, py_val, np_val, axpy_val, axrs_val):
        py_s = f'{py_val:.1f}' if py_val else '-'
        np_s = f'{np_val:.1f}'
        axpy_s = f'{axpy_val:.1f}'
        axrs_s = f'{axrs_val:.3f}' if axrs_val < 1 else f'{axrs_val:.1f}'
        ratio_py = f'{axpy_val/axrs_val:.0f}x' if axrs_val > 0 else '-'
        ratio_np = f'{np_val/axrs_val:.0f}x' if axrs_val > 0 else '-'
        return [name, str(dim), py_s, np_s, axpy_s, axrs_s, ratio_py, ratio_np]

    for i, d in enumerate(dims_born):
        rows.append(row('Born rule', d, py_born[i], np_born[i], axpy_born[i], axrs_born[i]))
    for i, d in enumerate(dims_cborn):
        rows.append(row('Complex Born', d, 0, np_cborn[i], axpy_cborn[i], axrs_cborn[i]))
    for i, d in enumerate(dims_mat):
        rows.append(row('Matmul', d, 0, np_mat[i], axpy_mat[i], axrs_mat[i]))
    for i, d in enumerate(dims_int):
        rows.append(row('Interference', d, 0, np_int[i], axpy_int[i], axrs_int[i]))
    for i, d in enumerate(dims_den):
        rows.append(row('Density+purity', d, 0, np_den[i], axpy_den[i], axrs_den[i]))
    for i, d in enumerate(dims_ch):
        rows.append(row('Dephasing', d, 0, np_ch[i], axpy_ch[i], axrs_ch[i]))

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#30363d')
        if r == 0:
            cell.set_facecolor('#1f6feb')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#c9d1d9')
        else:
            cell.set_facecolor('#0d1117')
            cell.set_text_props(color='#c9d1d9')
        # Highlight Rust column
        if c == 5 and r > 0:
            cell.set_text_props(color='#da3633', fontweight='bold')
        if c in (6, 7) and r > 0:
            cell.set_text_props(color='#3fb950', fontweight='bold')

    path = os.path.join(OUT_DIR, 'comparison_summary.svg')
    fig.tight_layout()
    fig.savefig(path, format='svg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')

    # ---- Print text summary ----
    print('\n' + '=' * 70)
    print('  Results Summary')
    print('=' * 70)
    print(f'\n  {"Op":<18} {"dim":>4} | {"Python":>10} {"NumPy":>10} {"AXOL-Py":>10} {"Rust":>10} | {"Rust/Py":>8} {"Rust/NP":>8}')
    print('  ' + '-' * 88)
    for r in rows:
        print(f'  {r[0]:<18} {r[1]:>4} | {r[2]:>10} {r[3]:>10} {r[4]:>10} {r[5]:>10} | {r[6]:>8} {r[7]:>8}')

    print(f'\n  SVGs saved to: {os.path.abspath(OUT_DIR)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
