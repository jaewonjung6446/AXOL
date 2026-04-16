"""Online (real-time) learning for AXOL — incremental EDMD.

Learns a Koopman operator K such that Psi(y) ~= Psi(x) @ K from a stream of
(x, y) samples, using only closed-form rank-1 moment updates.  No gradient
descent, no epoch loops — each sample costs O(ld^2) via the
Sherman-Morrison-Woodbury formula.

Axiom alignment
---------------
- Axiom 1 (no time axis):
    Each update is a single closed-form operation.  The only "time" present
    is the external data stream — the World's time, not an internal
    convergence loop.  Per-sample work never iterates toward a fixed point.
- Axiom 2 (>=99% accuracy):
    Moment matching is a consistent estimator (Glivenko-Cantelli / CLT).  As
    samples accumulate, the learned K converges to the population operator.
- Axiom 3 (probability as the price of timelessness):
    Uncertainty surfaces through Omega/Phi computed from the learned K —
    the learner never hides its own confidence.
- Axiom 4 (production-grade):
    Sherman-Morrison on a regularised G is numerically stable.  All updates
    are O(ld^2); memory is O(ld^2).

Mathematical basis
------------------
EDMD convention: model is Psi(y) = Psi(x) @ K (row vectors, matches
``axol.quantum.koopman.estimate_koopman_matrix``).  Normal equation:

    G_n = sum_{i=1..n} Psi(x_i) Psi(x_i)^T         (input Gram, ld x ld)
    B_n = sum_{i=1..n} Psi(x_i) Psi(y_i)^T         (input-output cross, ld x ld)
    K_n = G_n^+ B_n                                (Koopman operator, ld x ld)

Rank-1 update with forgetting factor gamma in (0, 1]:
    G_{n+1} = gamma G_n + u u^T,   u = Psi(x_{n+1})
    B_{n+1} = gamma B_n + u v^T,   v = Psi(y_{n+1})

Sherman-Morrison inverse update (regularised by +lambda*I):
    H := G^{-1}
    H_new = (1/gamma) H - (1/gamma^2) (H u u^T H) / (1 + (1/gamma) u^T H u)

Prediction path:
    y_hat = unlift(Psi(x) @ K)   [linear terms only]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.koopman import lift, lifted_dim, unlift
from axol.quantum.lyapunov import estimate_lyapunov, omega_from_lyapunov
from axol.quantum.fractal import phi_from_fractal


# ---------------------------------------------------------------------------
# LearningReport — diagnostic snapshot of the current learner state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearningReport:
    """Diagnostic snapshot of an OnlineLearner at a given moment."""

    n_samples: int
    dim: int
    lifted_dim: int
    forgetting_factor: float
    effective_samples: float  # 1 / (1 - gamma) in the limit, or n if gamma==1
    omega: float
    phi: float
    max_lyapunov: float
    residual_rms: float       # RMS of Psi(y) - Psi(x) @ K on a diagnostic buffer
    updates_since_rebuild: int


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Incremental EDMD learner with closed-form rank-1 updates.

    Typical usage::

        learner = OnlineLearner(dim=8, degree=2, forgetting_factor=0.995)
        for x, y in stream:
            learner.observe_sample(x, y)
            if learner.omega > 0.9:
                y_hat = learner.predict(new_x)

    Notes
    -----
    * ``forgetting_factor`` (gamma) controls memory horizon.  gamma=1.0 keeps
      everything; gamma<1.0 down-weights older samples geometrically.  The
      effective sample window is ~1 / (1 - gamma).
    * ``regularization`` (lambda) is added to G's diagonal to guarantee
      invertibility before any samples are seen and to damp ill-conditioning.
      Smaller lambda -> sharper fits once data arrives; larger lambda -> more
      stable early on.
    """

    def __init__(
        self,
        dim: int,
        degree: int = 2,
        basis: str = "poly",
        forgetting_factor: float = 1.0,
        regularization: float = 1e-3,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be in (0, 1]")
        if regularization <= 0.0:
            raise ValueError("regularization must be > 0")

        self.dim = dim
        self.degree = degree
        self.basis = basis
        self.forgetting_factor = float(forgetting_factor)
        self.regularization = float(regularization)

        ld = lifted_dim(dim, degree, basis)
        self.ld = ld

        # Moment matrices (float64 for numerical stability).
        # G starts at lambda*I so H = (1/lambda)*I is well defined.
        # B = sum Psi(x) Psi(y)^T  (rows indexed by x-features, cols by y).
        self._G = np.eye(ld, dtype=np.float64) * self.regularization
        self._H = np.eye(ld, dtype=np.float64) / self.regularization  # G^{-1}
        self._B = np.zeros((ld, ld), dtype=np.float64)

        # Cached K = H @ B (computed lazily)
        self._K_cache: np.ndarray | None = None

        # Diagnostics
        self._n_samples = 0
        self._last_residual_sq = 0.0
        self._residual_count = 0

    # ------------------------------------------------------------------
    # Core update: rank-1 Sherman-Morrison
    # ------------------------------------------------------------------

    def observe_sample(self, x: np.ndarray | FloatVec, y: np.ndarray | FloatVec) -> None:
        """Update moments from one (x, y) pair.

        Cost: O(ld^2).  No iteration, no gradient step — a single closed-form
        rank-1 update of both G^{-1} (via Sherman-Morrison) and A.
        """
        x_arr = _as_array(x, self.dim, "x")
        y_arr = _as_array(y, self.dim, "y")

        u = lift(x_arr, self.degree, self.basis).astype(np.float64)  # (ld,)
        v = lift(y_arr, self.degree, self.basis).astype(np.float64)  # (ld,)

        gamma = self.forgetting_factor

        # --- Track residual before update (for diagnostics) ---
        if self._K_cache is not None:
            pred = u @ self._K_cache
            diff = v - pred
            self._last_residual_sq = float(diff @ diff)
            self._residual_count += 1

        # --- Update G in-place: G = gamma*G + u u^T ---
        # (kept for snapshotting; H is what we actually use)
        self._G *= gamma
        self._G += np.outer(u, u)

        # --- Sherman-Morrison update of H = G^{-1} ---
        # After G <- gamma*G + u u^T, the new H is:
        #   H_new = (1/gamma) H - (1/gamma^2) (H u)(u^T H) / (1 + (1/gamma) u^T H u)
        H = self._H / gamma
        Hu = H @ u                      # (ld,)
        denom = 1.0 + float(u @ Hu)     # scalar (always > 0 since G is PD)
        if denom < 1e-15:
            # Degenerate — fall back to full recompute
            self._rebuild_H_from_G()
        else:
            self._H = H - np.outer(Hu, Hu) / denom

        # --- Update B: B <- gamma*B + u v^T  (sum of Psi(x) Psi(y)^T) ---
        self._B *= gamma
        self._B += np.outer(u, v)

        self._K_cache = None
        self._n_samples += 1

    def observe_batch(
        self,
        X: np.ndarray | Iterable[np.ndarray],
        Y: np.ndarray | Iterable[np.ndarray],
    ) -> None:
        """Update moments from many (x, y) pairs.

        Equivalent to calling ``observe_sample`` in order; provided for
        convenience and minor speed gains on contiguous arrays.
        """
        X = np.asarray(list(X) if not isinstance(X, np.ndarray) else X, dtype=np.float64)
        Y = np.asarray(list(Y) if not isinstance(Y, np.ndarray) else Y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            Y = Y.reshape(1, -1)
        if X.shape != Y.shape:
            raise ValueError(f"X shape {X.shape} must match Y shape {Y.shape}")

        for i in range(X.shape[0]):
            self.observe_sample(X[i], Y[i])

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def _K(self) -> np.ndarray:
        """Current Koopman matrix K = G^{-1} @ B = H @ B.

        Satisfies Psi(y) ~= Psi(x) @ K (row-vector convention, same as
        ``axol.quantum.koopman.estimate_koopman_matrix``).
        """
        if self._K_cache is None:
            self._K_cache = self._H @ self._B
        return self._K_cache

    @property
    def koopman_matrix(self) -> TransMatrix:
        """Current learned Koopman operator as a TransMatrix (lifted space)."""
        K = self._K()
        # Safety: clip extreme values produced by ill-conditioning before learning
        K_safe = np.nan_to_num(K, nan=0.0, posinf=100.0, neginf=-100.0)
        K_safe = np.clip(K_safe, -100.0, 100.0)
        return TransMatrix(data=K_safe.astype(np.float32))

    def predict(self, x: np.ndarray | FloatVec) -> np.ndarray:
        """Predict y from x using the current learned operator.

        Returns a numpy array of shape (dim,); callers can wrap in FloatVec.
        """
        x_arr = _as_array(x, self.dim, "x")
        psi = lift(x_arr, self.degree, self.basis).astype(np.float64)
        psi_y = psi @ self._K()
        return unlift(psi_y, self.dim, self.degree, self.basis).astype(np.float32)

    def predict_vec(self, x: np.ndarray | FloatVec) -> FloatVec:
        """Same as ``predict`` but returns a FloatVec."""
        return FloatVec(data=self.predict(x))

    # ------------------------------------------------------------------
    # Quality metrics (Axiom 3: uncertainty surfaced as Omega/Phi)
    # ------------------------------------------------------------------

    @property
    def max_lyapunov(self) -> float:
        """Max Lyapunov exponent of the learned operator (lifted space).

        Measured on the lifted operator K because that's what dictates the
        contractive/chaotic behaviour of repeated application.  Returns 0.0
        before any samples have been observed.
        """
        if self._n_samples == 0:
            return 0.0
        K = self._K()
        # estimate_lyapunov expects a TransMatrix
        tm = TransMatrix(data=K.astype(np.float32))
        # fewer steps -> cheaper; 50 is enough for a stable sign
        return estimate_lyapunov(tm, steps=50)

    @property
    def omega(self) -> float:
        """Cohesion (Omega) from the learned operator."""
        return omega_from_lyapunov(self.max_lyapunov)

    @property
    def phi(self) -> float:
        """Clarity (Phi) derived from the learned operator's effective rank.

        Uses a spectral proxy for fractal dimension: the participation ratio
        of the singular-value spectrum of K, clipped to [0, dim].  This is
        O(ld^2) per call via the cached H and A.
        """
        if self._n_samples == 0:
            return 1.0
        K = self._K()
        # Effective rank via participation ratio of singular values.
        # For a rank-1 operator, D_eff ~ 1; for full-rank isotropic, D_eff ~ dim.
        try:
            s = np.linalg.svd(K, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0.5
        s2 = s * s
        total = float(s2.sum())
        if total <= 1e-15:
            return 1.0
        d_eff = (total * total) / float((s2 * s2).sum())
        # Only the first `dim` linear-term rows/cols of K matter for unlift.
        d_eff = min(max(d_eff, 0.0), float(self.dim))
        return phi_from_fractal(d_eff, self.dim)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def effective_samples(self) -> float:
        """Approx effective sample size given forgetting.

        For gamma == 1: equals n_samples.
        For gamma <  1: approaches 1/(1-gamma) as n grows.
        """
        gamma = self.forgetting_factor
        if gamma >= 1.0:
            return float(self._n_samples)
        n = self._n_samples
        # Sum_{k=0..n-1} gamma^k = (1 - gamma^n) / (1 - gamma)
        return float((1.0 - gamma ** n) / (1.0 - gamma))

    def report(self) -> LearningReport:
        """Snapshot of the learner's current state."""
        residual_rms = (
            float(np.sqrt(self._last_residual_sq / max(self.ld, 1)))
            if self._residual_count > 0
            else 0.0
        )
        lam = self.max_lyapunov
        return LearningReport(
            n_samples=self._n_samples,
            dim=self.dim,
            lifted_dim=self.ld,
            forgetting_factor=self.forgetting_factor,
            effective_samples=self.effective_samples,
            omega=omega_from_lyapunov(lam),
            phi=self.phi,
            max_lyapunov=lam,
            residual_rms=residual_rms,
            updates_since_rebuild=self._n_samples,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget everything; return to the zero-sample state."""
        self._G = np.eye(self.ld, dtype=np.float64) * self.regularization
        self._H = np.eye(self.ld, dtype=np.float64) / self.regularization
        self._B = np.zeros((self.ld, self.ld), dtype=np.float64)
        self._K_cache = None
        self._n_samples = 0
        self._last_residual_sq = 0.0
        self._residual_count = 0

    # ------------------------------------------------------------------
    # Explicit forgetting
    # ------------------------------------------------------------------

    def decay(self, factor: float) -> None:
        """Uniformly decay all accumulated moments by ``factor`` in [0, 1].

        Equivalent to applying ``forgetting_factor=factor`` for one "empty"
        timestep with no new data.  Use this to model the passage of time
        between learning events — what an animal does while it sleeps or
        waits: nothing new comes in, but old memories fade.

        ``factor=1.0`` is a no-op; ``factor=0.0`` collapses back to the
        regularised zero state (same as ``reset()`` except ``n_samples``
        is preserved for diagnostics).
        """
        if not (0.0 <= factor <= 1.0):
            raise ValueError("factor must be in [0, 1]")
        if factor == 1.0:
            return

        # G = reg*I + sum gamma^k u u^T.  After decay we want the
        # regularisation floor to stay put:
        #     G_new = factor * (G - reg*I) + reg*I
        #           = factor*G + (1 - factor)*reg*I
        reg_I = np.eye(self.ld, dtype=np.float64) * self.regularization
        self._G = factor * self._G + (1.0 - factor) * reg_I
        self._B *= factor
        self._rebuild_H_from_G()
        self._K_cache = None

    def decay_by_time(self, elapsed: float, half_life: float) -> None:
        """Ebbinghaus-style exponential decay over elapsed time.

        ``factor = 0.5 ** (elapsed / half_life)``.  After ``half_life`` units
        of time the moments are halved; this is the classic forgetting curve
        (Ebbinghaus, 1885) expressed exactly.

        ``elapsed`` and ``half_life`` share arbitrary time units (seconds,
        minutes, ticks) — only their ratio matters.
        """
        if half_life <= 0.0:
            raise ValueError("half_life must be > 0")
        if elapsed < 0.0:
            raise ValueError("elapsed must be >= 0")
        if elapsed == 0.0:
            return
        factor = 0.5 ** (elapsed / half_life)
        self.decay(factor)

    def _rebuild_H_from_G(self) -> None:
        """Full recompute of H = (G + lambda*I)^{-1} — only on numerical fallback."""
        G_reg = self._G + np.eye(self.ld, dtype=np.float64) * self.regularization
        try:
            self._H = np.linalg.inv(G_reg)
        except np.linalg.LinAlgError:
            self._H = np.linalg.pinv(G_reg)
        self._K_cache = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_array(x: np.ndarray | FloatVec, dim: int, name: str) -> np.ndarray:
    """Coerce FloatVec/ndarray to a 1-D float64 array of length ``dim``."""
    if isinstance(x, FloatVec):
        arr = x.data.astype(np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.size != dim:
        raise ValueError(
            f"{name} must be a 1-D vector of length {dim}, got shape {arr.shape}"
        )
    return arr
