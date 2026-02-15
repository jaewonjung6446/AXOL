//! Linear Readout Layer — the only trainable component in WTE.
//!
//! Maps the reservoir's feature vector to task-specific outputs via
//! a simple linear projection trained with least squares (lstsq).
//!
//! Feature vector structure:
//!   [merged(dim) + scale_0(dim) + scale_1(dim) + scale_2(dim) + coherence(1) + energy(1)]
//!   = 4*dim + 2 dimensions for 3 scales (e.g., 258 for dim=64)
//!
//! Training: single-shot lstsq — no SGD, no epochs, no hyperparameters.

use super::reservoir::ReservoirState;

// ---------------------------------------------------------------------------
// LinearReadout — trainable output projection
// ---------------------------------------------------------------------------

/// Linear readout layer: y = W × x + b.
///
/// The only trained component in the WTE pipeline.
/// Training uses least squares (lstsq) for a one-shot solution.
#[derive(Clone, Debug)]
pub struct LinearReadout {
    /// Input feature dimension (4*dim + 2 for 3 scales)
    pub feature_dim: usize,
    /// Output dimension (vocab_size for autocomplete, n_classes for classification)
    pub output_dim: usize,
    /// Weight matrix: output_dim × feature_dim (row-major)
    pub weights: Vec<f64>,
    /// Bias vector: output_dim
    pub bias: Vec<f64>,
    /// Whether this readout has been trained
    pub trained: bool,
}

impl LinearReadout {
    /// Create a new untrained readout layer.
    pub fn new(feature_dim: usize, output_dim: usize) -> Self {
        Self {
            feature_dim,
            output_dim,
            weights: vec![0.0; output_dim * feature_dim],
            bias: vec![0.0; output_dim],
            trained: false,
        }
    }

    /// Forward pass: compute output scores from a feature vector.
    ///
    /// Returns raw scores (logits). Apply softmax for probabilities.
    pub fn forward(&self, features: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_dim];
        let feat_len = features.len().min(self.feature_dim);

        for i in 0..self.output_dim {
            let mut sum = self.bias[i];
            for j in 0..feat_len {
                sum += self.weights[i * self.feature_dim + j] * features[j];
            }
            output[i] = sum;
        }

        output
    }

    /// Train the readout using least squares.
    ///
    /// `features_batch`: slice of feature vectors (one per sample)
    /// `targets_batch`: slice of target vectors (one per sample)
    ///
    /// Solves W = argmin_W Σ ||W × x_i - y_i||² via normal equations.
    pub fn train(&mut self, features_batch: &[Vec<f64>], targets_batch: &[Vec<f64>]) {
        let n = features_batch.len();
        if n == 0 || targets_batch.len() != n {
            return;
        }

        let fd = self.feature_dim;
        let od = self.output_dim;

        // Compute mean target for bias initialization
        let mut mean_target = vec![0.0; od];
        for target in targets_batch {
            for j in 0..od.min(target.len()) {
                mean_target[j] += target[j];
            }
        }
        for v in mean_target.iter_mut() {
            *v /= n as f64;
        }

        // XtX = X^T @ X (fd × fd)
        let mut xtx = vec![0.0f64; fd * fd];
        for i in 0..fd {
            for j in 0..fd {
                let mut sum = 0.0;
                for k in 0..n {
                    let xi = features_batch[k].get(i).copied().unwrap_or(0.0);
                    let xj = features_batch[k].get(j).copied().unwrap_or(0.0);
                    sum += xi * xj;
                }
                xtx[i * fd + j] = sum;
            }
        }

        // Adaptive regularization: λ = 0.1% of average diagonal
        let trace: f64 = (0..fd).map(|i| xtx[i * fd + i]).sum::<f64>();
        let avg_diag = trace / fd as f64;
        let lambda = (avg_diag * 0.001).max(1e-6);
        for i in 0..fd {
            xtx[i * fd + i] += lambda;
        }

        // Invert XtX
        let xtx_inv = match invert_matrix(&xtx, fd) {
            Some(inv) => inv,
            None => return, // singular, skip
        };

        // For each output dimension, solve independently
        for o in 0..od {
            // XtY_o = X^T @ y[:,o] (fd × 1)
            let mut xty = vec![0.0f64; fd];
            for i in 0..fd {
                let mut sum = 0.0;
                for k in 0..n {
                    let xi = features_batch[k].get(i).copied().unwrap_or(0.0);
                    let yo = targets_batch[k].get(o).copied().unwrap_or(0.0);
                    sum += xi * yo;
                }
                xty[i] = sum;
            }

            // w_o = (XtX)^{-1} @ XtY_o
            for j in 0..fd {
                let mut val = 0.0;
                for i in 0..fd {
                    val += xtx_inv[j * fd + i] * xty[i];
                }
                self.weights[o * fd + j] = val;
            }

            self.bias[o] = mean_target[o];
        }

        self.trained = true;
    }

    /// Model size in bytes.
    pub fn size_bytes(&self) -> usize {
        (self.weights.len() + self.bias.len()) * 8
    }
}

// ---------------------------------------------------------------------------
// MlpReadout — nonlinear readout with hidden layer
// ---------------------------------------------------------------------------

/// MLP readout: features → hidden(tanh) → output.
///
/// Provides nonlinear feature transformation before the output projection.
/// Training: lstsq warm-start on output layer, then SGD fine-tuning.
#[derive(Clone, Debug)]
pub struct MlpReadout {
    pub feature_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    /// Hidden weights: hidden_dim × feature_dim (row-major)
    pub w_hidden: Vec<f64>,
    /// Hidden bias: hidden_dim
    pub b_hidden: Vec<f64>,
    /// Output weights: output_dim × hidden_dim (row-major)
    pub w_output: Vec<f64>,
    /// Output bias: output_dim
    pub b_output: Vec<f64>,
    pub trained: bool,
}

impl MlpReadout {
    /// Create with Xavier initialization.
    pub fn new(feature_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng: u64 = 42;
        let limit_h = (6.0 / (feature_dim + hidden_dim) as f64).sqrt();
        let w_hidden: Vec<f64> = (0..hidden_dim * feature_dim)
            .map(|_| lcg_f64(&mut rng) * limit_h)
            .collect();
        let limit_o = (6.0 / (hidden_dim + output_dim) as f64).sqrt();
        let w_output: Vec<f64> = (0..output_dim * hidden_dim)
            .map(|_| lcg_f64(&mut rng) * limit_o * 0.01)
            .collect();

        Self {
            feature_dim, hidden_dim, output_dim,
            w_hidden,
            b_hidden: vec![0.0; hidden_dim],
            w_output,
            b_output: vec![0.0; output_dim],
            trained: false,
        }
    }

    /// Forward through hidden layer only (tanh activation).
    fn forward_hidden(&self, features: &[f64]) -> Vec<f64> {
        let fl = features.len().min(self.feature_dim);
        let mut h = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut s = self.b_hidden[i];
            let row = i * self.feature_dim;
            for j in 0..fl {
                s += self.w_hidden[row + j] * features[j];
            }
            h[i] = s.tanh();
        }
        h
    }

    /// Full forward pass: features → hidden(tanh) → output.
    pub fn forward(&self, features: &[f64]) -> Vec<f64> {
        let h = self.forward_hidden(features);
        let mut out = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut s = self.b_output[i];
            let row = i * self.hidden_dim;
            for j in 0..self.hidden_dim {
                s += self.w_output[row + j] * h[j];
            }
            out[i] = s;
        }
        out
    }

    /// Train via ELM: random hidden layer (fixed) + lstsq output layer.
    ///
    /// The hidden layer provides nonlinear feature expansion via tanh,
    /// while the output layer is solved optimally via least squares.
    /// No SGD needed — this is guaranteed to find the global optimum
    /// for the output weights given the fixed hidden representation.
    pub fn train(&mut self, features_batch: &[Vec<f64>], targets_batch: &[Vec<f64>]) {
        let n = features_batch.len();
        if n == 0 || targets_batch.len() != n { return; }

        // Forward through fixed random hidden layer
        let hidden_batch: Vec<Vec<f64>> = features_batch.iter()
            .map(|f| self.forward_hidden(f))
            .collect();

        // Optimal output layer via lstsq
        let mut out_linear = LinearReadout::new(self.hidden_dim, self.output_dim);
        out_linear.train(&hidden_batch, targets_batch);
        if out_linear.trained {
            self.w_output = out_linear.weights;
            self.b_output = out_linear.bias;
        }

        self.trained = true;
    }

    pub fn size_bytes(&self) -> usize {
        (self.w_hidden.len() + self.b_hidden.len()
            + self.w_output.len() + self.b_output.len()) * 8
    }
}

/// LCG pseudo-random in [-1, 1].
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// ReadoutLayer — unified enum for Linear / MLP readout
// ---------------------------------------------------------------------------

/// Readout layer: either Linear (lstsq) or MLP (hidden + SGD).
///
/// `hidden_dim == 0` → Linear, `hidden_dim > 0` → MLP.
#[derive(Clone, Debug)]
pub enum ReadoutLayer {
    Linear(LinearReadout),
    Mlp(MlpReadout),
}

impl ReadoutLayer {
    pub fn new(feature_dim: usize, output_dim: usize, hidden_dim: usize) -> Self {
        if hidden_dim == 0 {
            ReadoutLayer::Linear(LinearReadout::new(feature_dim, output_dim))
        } else {
            ReadoutLayer::Mlp(MlpReadout::new(feature_dim, hidden_dim, output_dim))
        }
    }

    pub fn forward(&self, features: &[f64]) -> Vec<f64> {
        match self { ReadoutLayer::Linear(r) => r.forward(features), ReadoutLayer::Mlp(r) => r.forward(features) }
    }

    pub fn train(&mut self, features_batch: &[Vec<f64>], targets_batch: &[Vec<f64>]) {
        match self { ReadoutLayer::Linear(r) => r.train(features_batch, targets_batch), ReadoutLayer::Mlp(r) => r.train(features_batch, targets_batch) }
    }

    pub fn trained(&self) -> bool {
        match self { ReadoutLayer::Linear(r) => r.trained, ReadoutLayer::Mlp(r) => r.trained }
    }

    pub fn size_bytes(&self) -> usize {
        match self { ReadoutLayer::Linear(r) => r.size_bytes(), ReadoutLayer::Mlp(r) => r.size_bytes() }
    }

    pub fn feature_dim(&self) -> usize {
        match self { ReadoutLayer::Linear(r) => r.feature_dim, ReadoutLayer::Mlp(r) => r.feature_dim }
    }

    pub fn output_dim(&self) -> usize {
        match self { ReadoutLayer::Linear(r) => r.output_dim, ReadoutLayer::Mlp(r) => r.output_dim }
    }

    pub fn type_tag(&self) -> u8 {
        match self { ReadoutLayer::Linear(_) => 0, ReadoutLayer::Mlp(_) => 1 }
    }
}

// ---------------------------------------------------------------------------
// Feature extraction helpers
// ---------------------------------------------------------------------------

/// Extract features from a ReservoirState for the readout layer.
pub fn extract_features(state: &ReservoirState) -> Vec<f64> {
    state.to_feature_vector()
}

// ---------------------------------------------------------------------------
// Matrix inversion (Gauss-Jordan)
// ---------------------------------------------------------------------------

/// Invert a dim×dim matrix using Gauss-Jordan elimination.
fn invert_matrix(a: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0f64; dim * 2 * dim];

    // [A | I]
    for i in 0..dim {
        for j in 0..dim {
            aug[i * 2 * dim + j] = a[i * dim + j];
        }
        aug[i * 2 * dim + dim + i] = 1.0;
    }

    let stride = 2 * dim;

    for col in 0..dim {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * stride + col].abs();
        for row in (col + 1)..dim {
            let val = aug[row * stride + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }

        if max_row != col {
            for j in 0..stride {
                let tmp = aug[col * stride + j];
                aug[col * stride + j] = aug[max_row * stride + j];
                aug[max_row * stride + j] = tmp;
            }
        }

        let pivot = aug[col * stride + col];
        for j in 0..stride {
            aug[col * stride + j] /= pivot;
        }

        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row * stride + col];
            for j in 0..stride {
                aug[row * stride + j] -= factor * aug[col * stride + j];
            }
        }
    }

    let mut inv = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            inv[i * dim + j] = aug[i * stride + dim + j];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readout_forward() {
        let mut readout = LinearReadout::new(4, 2);
        // Set weights manually
        readout.weights = vec![
            1.0, 0.0, 0.0, 0.0, // output 0 = feature 0
            0.0, 1.0, 0.0, 0.0, // output 1 = feature 1
        ];
        readout.bias = vec![0.1, 0.2];

        let features = vec![0.5, 0.3, 0.1, 0.1];
        let output = readout.forward(&features);
        assert!((output[0] - 0.6).abs() < 1e-6, "output[0] = {}", output[0]);
        assert!((output[1] - 0.5).abs() < 1e-6, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_readout_train() {
        let mut readout = LinearReadout::new(3, 2);

        // Simple training data: output = 2*x0, 3*x1
        let features = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];
        let targets = vec![
            vec![2.0, 0.0],
            vec![0.0, 3.0],
            vec![2.0, 3.0],
            vec![1.0, 1.5],
        ];

        readout.train(&features, &targets);
        assert!(readout.trained);

        // Test prediction
        let pred = readout.forward(&vec![1.0, 0.0, 0.0]);
        assert!((pred[0] - 2.0).abs() < 2.0, "pred[0] = {}", pred[0]);
    }

    #[test]
    fn test_invert_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let inv = invert_matrix(&a, 2).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i * 2 + j] - expected).abs() < 1e-6);
            }
        }
    }
}
