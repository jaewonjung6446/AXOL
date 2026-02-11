# AXOL: Chaos Theory Formalization

## 0. Overview

This document formalizes the AXOL paradigm using **chaos theory** as the mathematical backbone.
THEORY.md introduced the conceptual model (Declare -> Weave -> Observe).
This document provides the rigorous mathematical definitions.

**Core Mapping:**

| AXOL Concept | Chaos Theory Concept | Mathematical Object |
|---|---|---|
| Tapestry | Strange Attractor | Compact invariant set in phase space |
| Omega (Cohesion) | Lyapunov Stability | `1/(1 + max(lambda, 0))` |
| Phi (Clarity) | Fractal Dimension Inverse | `1/(1 + D/D_max)` |
| Weave | Attractor Construction | Iterative map building |
| Observe | Point Collapse on Attractor | Single-point sampling, O(D) |
| Entanglement Range | Basin of Attraction | Boundary of convergence region |

---

## 1. Phase Space and Attractors

### 1.1 Phase Space

An AXOL program operates in a **phase space** R^n where n is determined by the declared inputs and outputs.

Each declared variable corresponds to a dimension (or set of dimensions) in this space.
The state of the system at any point is a vector `x in R^n`.

### 1.2 Strange Attractor (Tapestry)

A **Tapestry** is mathematically a strange attractor:

> A compact subset `A` of phase space `R^n` such that:
> 1. `A` is invariant under the system dynamics `f`: `f(A) = A`
> 2. `A` has a basin of attraction `B(A)` with non-zero measure
> 3. `A` has sensitive dependence on initial conditions (chaos)
> 4. `A` has non-integer (fractal) dimension

The weaving process constructs the map `f: R^n -> R^n` whose attractor
encodes the declared relationships.

### 1.3 Trajectory Matrix

The dynamics are represented by a **trajectory matrix** `M in R^(n x n)`:

```
x_{k+1} = x_k @ M + perturbation
```

For the linearized system, `M` is a TransMatrix from axol.core.
Non-linear effects are captured through iterative application and interference layers.

---

## 2. Lyapunov Exponents and Cohesion (Omega)

### 2.1 Lyapunov Exponent

The **maximum Lyapunov exponent** `lambda` quantifies sensitivity to initial conditions:

```
lambda = lim_{k->inf} (1/k) * ln(||delta x_k|| / ||delta x_0||)
```

- `lambda < 0`: trajectories converge (stable fixed point/cycle)
- `lambda = 0`: neutral stability (limit cycle boundary)
- `lambda > 0`: trajectories diverge (chaos)

### 2.2 Lyapunov Spectrum

For an n-dimensional system, there are n Lyapunov exponents `lambda_1 >= lambda_2 >= ... >= lambda_n`.

The **Benettin method** estimates these via QR decomposition of the tangent map:

```
Algorithm: Benettin's QR method
1. Initialize orthonormal vectors {e_1, ..., e_n}
2. For each time step k:
   a. Propagate: e_i' = M @ e_i
   b. QR decompose: [e_1'|...|e_n'] = Q @ R
   c. Accumulate: lambda_i += ln(R_ii)
   d. Set e_i = column_i(Q)
3. lambda_i = (1/K) * accumulated_lambda_i
```

### 2.3 Cohesion from Lyapunov

**Omega (Cohesion)** measures observation stability:

```
Omega = 1 / (1 + max(lambda, 0))
```

Properties:
- `lambda << 0` => `Omega -> 1.0` (strong convergence, stable observations)
- `lambda = 0` => `Omega = 1.0` (marginally stable)
- `lambda = 1` => `Omega = 0.5` (moderate chaos)
- `lambda -> inf` => `Omega -> 0.0` (full chaos, random observations)

**Empirical Omega** from repeated observations:

```
Omega_empirical = (count of mode == argmax) / total_observations
```

---

## 3. Fractal Dimension and Clarity (Phi)

### 3.1 Fractal Dimension

The **fractal dimension** `D` of an attractor quantifies its geometric complexity.

**Box-counting dimension:**

```
D = lim_{eps->0} ln(N(eps)) / ln(1/eps)
```

where `N(eps)` is the number of boxes of side length `eps` needed to cover the attractor.

**Correlation dimension** (Grassberger-Procaccia):

```
C(r) = lim_{N->inf} (2 / N(N-1)) * sum_{i<j} Theta(r - ||x_i - x_j||)
D_corr = lim_{r->0} ln(C(r)) / ln(r)
```

### 3.2 Clarity from Fractal Dimension

**Phi (Clarity)** measures output precision:

```
Phi = 1 / (1 + D / D_max)
```

where `D_max = n` (phase space dimension) is the maximum possible dimension.

Properties:
- `D = 0` (fixed point) => `Phi = 1.0` (perfectly precise)
- `D = D_max` (space-filling) => `Phi = 0.5`
- `D >> D_max` (impossible but as limit) => `Phi -> 0.0`

**Entropy-based Phi** (approximation from probability distribution):

```
H = -sum_i p_i * ln(p_i)     (Shannon entropy)
H_max = ln(n)
Phi_entropy = 1 - H / H_max
```

---

## 4. Entanglement Cost (E)

### 4.1 Cost Model

The entanglement cost `E` represents the computational effort to construct the attractor:

```
E = sum_path [ iterations_to_converge(path) * path_complexity(path) ]
```

where:
- **iterations_to_converge**: determined by Lyapunov exponent of the path
  - `lambda << 0`: fast convergence, ~O(1/|lambda|) iterations
  - `lambda ~ 0`: slow convergence, ~O(1/epsilon) iterations
  - `lambda > 0`: no convergence, cost is unbounded (warning issued)

- **path_complexity**: number of interference layers needed
  - `complexity = ceil(log2(budget / BASE_COST))`

### 4.2 Achievability

A quality target `(Omega_target, Phi_target)` is **achievable** iff:

```
Omega_target <= 1 / (1 + max(lambda_max_path, 0))
Phi_target <= 1 / (1 + D_estimated / D_max)
```

If not achievable, the weaver reports:
- Maximum achievable Omega and Phi
- The critical path causing the limitation
- The Lyapunov exponent of that path

---

## 5. Composition Rules

### 5.1 Serial Composition

For two stages A and B in series:

```
lambda_total = lambda_A + lambda_B
Omega_total = 1 / (1 + max(lambda_A + lambda_B, 0))
D_total <= D_A + D_B
Phi_total >= Phi_A * Phi_B
```

Intuition: serial composition accumulates chaos. Two mildly chaotic stages
can produce a strongly chaotic pipeline.

### 5.2 Parallel Composition

For two stages A and B in parallel:

```
lambda_total = max(lambda_A, lambda_B)
Omega_total = min(Omega_A, Omega_B)
D_total = max(D_A, D_B)
Phi_total = min(Phi_A, Phi_B)
```

Intuition: parallel composition is limited by the weakest link.

### 5.3 Observation and Reuse

After observation:
- If `lambda < 0` (convergent): tapestry can be reused without re-weaving
- If `lambda > 0` (chaotic): tapestry must be re-woven (observation perturbs state)

---

## 6. Observation Complexity

### 6.1 Single Observation

Observation collapses the system state to a single point on the attractor.

**Time complexity**: `O(D)` where `D` is the attractor's embedding dimension.

This is because:
1. The attractor lives in a D-dimensional subspace of R^n
2. Projection onto this subspace is O(D)
3. Born rule application is O(D)

### 6.2 Re-observation

Multiple observations improve quality:

```
Omega_k >= 1 - (1 - Omega_1)^k    (convergence of mode stability)
```

where k is the number of observations.

---

## 7. Validation Benchmarks

### 7.1 Lorenz System

The Lorenz attractor serves as a known benchmark:
- `lambda_max ~ 0.91` => `Omega ~ 0.52`
- `D ~ 2.06` => `Phi ~ 1/(1 + 2.06/3) ~ 0.59` (for 3D phase space)

### 7.2 Convergent System

A system with `lambda < 0`:
- `lambda = -2.0` => `Omega = 1.0`
- `D ~ 0` (fixed point) => `Phi ~ 1.0`

### 7.3 Uniform Distribution

- Shannon entropy `H = H_max` => `Phi_entropy = 0.0`
- Fractal dim `D = D_max` => `Phi = 0.5`

### 7.4 Delta Distribution

- Shannon entropy `H = 0` => `Phi_entropy = 1.0`
- Fractal dim `D = 0` => `Phi = 1.0`
