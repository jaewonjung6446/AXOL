"""Fractal dimension estimation and Phi (clarity) calculation.

Mathematical basis:
  D = lim_{eps->0} ln(N(eps)) / ln(1/eps)       (box-counting)
  Phi = 1 / (1 + D / D_max)
"""

from __future__ import annotations

import numpy as np

from axol.core.types import FloatVec


def estimate_fractal_dim(
    attractor_points: FloatVec,
    method: str = "box_counting",
    phase_space_dim: int | None = None,
) -> float:
    """Estimate the fractal dimension of attractor points.

    Args:
        attractor_points: Flattened array of attractor trajectory points.
            If phase_space_dim is given, reshaped to (N, phase_space_dim).
        method: "box_counting" or "correlation".
        phase_space_dim: Dimension of the phase space for reshaping.

    Returns:
        Estimated fractal dimension D >= 0.
    """
    data = attractor_points.data.astype(np.float64)

    if data.size == 0:
        return 0.0

    # Reshape if we know the phase space dimension
    if phase_space_dim is not None and phase_space_dim > 0:
        n_points = data.size // phase_space_dim
        if n_points < 2:
            return 0.0
        data = data[: n_points * phase_space_dim].reshape(n_points, phase_space_dim)
    else:
        if data.ndim == 1:
            data = data.reshape(-1, 1)

    n_points = data.shape[0]
    if n_points < 2:
        return 0.0

    if method == "box_counting":
        return _box_counting_dim(data)
    elif method == "correlation":
        return _correlation_dim(data)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'box_counting' or 'correlation'.")


def _box_counting_dim(data: np.ndarray) -> float:
    """Box-counting dimension estimation."""
    n_points, n_dim = data.shape

    # Normalise to [0, 1] range
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalised = (data - mins) / ranges

    # Try different box sizes (eps)
    eps_values = []
    n_boxes_values = []

    for k in range(1, 8):
        n_grid = 2 ** k
        eps = 1.0 / n_grid
        # Assign each point to a grid cell
        cells = np.floor(normalised * (n_grid - 1e-10)).astype(np.int64)
        # Count unique occupied cells
        unique_cells = set(map(tuple, cells))
        n_boxes = len(unique_cells)

        if n_boxes > 0 and eps > 0:
            eps_values.append(np.log(1.0 / eps))
            n_boxes_values.append(np.log(n_boxes))

    if len(eps_values) < 2:
        return 0.0

    # Linear regression: ln(N) = D * ln(1/eps) + c
    x = np.array(eps_values)
    y = np.array(n_boxes_values)
    slope, _ = np.polyfit(x, y, 1)

    return max(float(slope), 0.0)


def _correlation_dim(data: np.ndarray) -> float:
    """Correlation dimension estimation (Grassberger-Procaccia)."""
    n_points = data.shape[0]

    if n_points < 10:
        return 0.0

    # Subsample if too many points
    max_sample = min(n_points, 500)
    rng = np.random.default_rng(42)
    indices = rng.choice(n_points, size=max_sample, replace=False) if n_points > max_sample else np.arange(n_points)
    sample = data[indices]

    # Compute pairwise distances
    dists = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            d = np.linalg.norm(sample[i] - sample[j])
            if d > 0:
                dists.append(d)

    if len(dists) < 10:
        return 0.0

    dists = np.array(dists)
    n_pairs = len(dists)

    # Correlation integral at various radii
    r_min = np.percentile(dists, 5)
    r_max = np.percentile(dists, 95)
    if r_min <= 0 or r_max <= r_min:
        return 0.0

    log_r = []
    log_c = []
    for k in range(10):
        r = r_min * ((r_max / r_min) ** (k / 9.0))
        c = np.sum(dists < r) / n_pairs
        if c > 0 and r > 0:
            log_r.append(np.log(r))
            log_c.append(np.log(c))

    if len(log_r) < 2:
        return 0.0

    slope, _ = np.polyfit(log_r, log_c, 1)
    return max(float(slope), 0.0)


def phi_from_fractal(fractal_dim: float, phase_space_dim: int) -> float:
    """Compute Phi (clarity) from fractal dimension.

    Phi = 1 / (1 + D / D_max)

    - D = 0 (fixed point)        => Phi = 1.0
    - D = D_max (space-filling)  => Phi = 0.5
    """
    if phase_space_dim <= 0:
        return 1.0
    return 1.0 / (1.0 + fractal_dim / phase_space_dim)


def phi_from_entropy(probs: FloatVec) -> float:
    """Compute Phi from Shannon entropy of a probability distribution.

    Phi = 1 - H / H_max

    - Delta distribution (one hot)  => H = 0     => Phi = 1.0
    - Uniform distribution           => H = H_max => Phi = 0.0
    """
    p = probs.data.astype(np.float64)
    p = p[p > 0]  # ignore zero entries

    if len(p) == 0:
        return 1.0

    n = probs.size
    if n <= 1:
        return 1.0

    h = -np.sum(p * np.log(p))
    h_max = np.log(n)

    if h_max <= 0:
        return 1.0

    phi = 1.0 - float(h / h_max)
    return max(0.0, min(1.0, phi))
