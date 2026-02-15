//! Semantic Phase Space (SPS) — Layer 1 of the AXOL Text Model.
//!
//! Maps vocabulary tokens to basin centroids in a chaotic phase space.
//! Each token becomes a Wave that can participate in interference.
//!
//! Key ideas:
//!   - Each token = a basin in dim-dimensional phase space
//!   - Similar tokens → nearby basins (via co-occurrence embeddings)
//!   - Token frequency → basin volume (Zipf's law ↔ basin volume distribution)
//!   - Token → Wave conversion uses basin proximity (soft assignment)

use num_complex::Complex64;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::types::*;
use crate::wave::Wave;
use crate::collapse::CollapseMetrics;
use crate::errors::Result;
use super::tokenizer::Vocabulary;

// ---------------------------------------------------------------------------
// SemanticPhaseSpace
// ---------------------------------------------------------------------------

/// The semantic phase space: a mapping from token IDs to waves.
///
/// Internally holds:
///   - Token embeddings (dim-dimensional vectors per token)
///   - A transformation matrix for context mixing
///   - Basin structure for quality metrics
#[derive(Clone, Debug)]
pub struct SemanticPhaseSpace {
    /// Dimension of the phase space
    pub dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding matrix: vocab_size × dim (each row = token embedding)
    pub embeddings: Vec<Vec<f64>>,
    /// Transformation matrix for wave processing (dim × dim)
    pub transform: TransMatrix,
}

impl SemanticPhaseSpace {
    /// Build SPS from corpus co-occurrence statistics.
    ///
    /// Process:
    ///   1. Build co-occurrence matrix from corpus
    ///   2. Reduce to dim dimensions via truncated SVD (power iteration)
    ///   3. Normalize embeddings
    ///   4. Build transformation matrix from embedding structure
    pub fn from_corpus(vocab: &Vocabulary, sentences: &[&str], dim: usize, seed: u64) -> Self {
        let vocab_size = vocab.size;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // --- Step 1: Co-occurrence matrix ---
        let window = 5;
        let mut cooccur = vec![0.0f64; vocab_size * vocab_size];

        for sentence in sentences {
            let ids = vocab.encode(sentence);
            for (i, &id_a) in ids.iter().enumerate() {
                for j in (i + 1)..ids.len().min(i + 1 + window) {
                    let id_b = ids[j];
                    let weight = 1.0 / (j - i) as f64; // distance decay
                    cooccur[id_a * vocab_size + id_b] += weight;
                    cooccur[id_b * vocab_size + id_a] += weight;
                }
            }
        }

        // PPMI transform: PMI(w1,w2) = log(P(w1,w2) / (P(w1)*P(w2))), clamped to 0
        let total: f64 = cooccur.iter().sum();
        if total > 1e-15 {
            // Compute marginals P(w)
            let mut marginal = vec![0.0f64; vocab_size];
            for i in 0..vocab_size {
                for j in 0..vocab_size {
                    marginal[i] += cooccur[i * vocab_size + j];
                }
                marginal[i] /= total;
            }

            // PPMI = max(0, log(P(w1,w2) / (P(w1)*P(w2))))
            for i in 0..vocab_size {
                for j in 0..vocab_size {
                    let p_ij = cooccur[i * vocab_size + j] / total;
                    let p_i = marginal[i];
                    let p_j = marginal[j];
                    if p_ij > 1e-15 && p_i > 1e-15 && p_j > 1e-15 {
                        cooccur[i * vocab_size + j] = (p_ij / (p_i * p_j)).ln().max(0.0);
                    } else {
                        cooccur[i * vocab_size + j] = 0.0;
                    }
                }
            }
        }

        // --- Step 2: Truncated SVD via power iteration ---
        let embeddings = truncated_svd(&cooccur, vocab_size, dim, 20, &mut rng);

        // --- Step 3: Build transformation matrix ---
        // Initialize with structure derived from embedding correlations
        let transform = build_transform(&embeddings, dim, &mut rng);

        Self {
            dim,
            vocab_size,
            embeddings,
            transform,
        }
    }

    /// Build SPS from pre-encoded token sequences (for use with BPE tokenizer).
    ///
    /// Instead of calling vocab.encode() on raw sentences, this method takes
    /// token ID sequences directly — allowing BPE-encoded inputs.
    pub fn from_encoded_corpus(
        vocab_size: usize,
        encoded_sentences: &[Vec<usize>],
        dim: usize,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Co-occurrence matrix from pre-encoded sequences
        let window = 5;
        let mut cooccur = vec![0.0f64; vocab_size * vocab_size];

        for ids in encoded_sentences {
            for (i, &id_a) in ids.iter().enumerate() {
                if id_a >= vocab_size {
                    continue;
                }
                for j in (i + 1)..ids.len().min(i + 1 + window) {
                    let id_b = ids[j];
                    if id_b >= vocab_size {
                        continue;
                    }
                    let weight = 1.0 / (j - i) as f64;
                    cooccur[id_a * vocab_size + id_b] += weight;
                    cooccur[id_b * vocab_size + id_a] += weight;
                }
            }
        }

        // PPMI transform
        let total: f64 = cooccur.iter().sum();
        if total > 1e-15 {
            let mut marginal = vec![0.0f64; vocab_size];
            for i in 0..vocab_size {
                for j in 0..vocab_size {
                    marginal[i] += cooccur[i * vocab_size + j];
                }
                marginal[i] /= total;
            }

            for i in 0..vocab_size {
                for j in 0..vocab_size {
                    let p_ij = cooccur[i * vocab_size + j] / total;
                    let p_i = marginal[i];
                    let p_j = marginal[j];
                    if p_ij > 1e-15 && p_i > 1e-15 && p_j > 1e-15 {
                        cooccur[i * vocab_size + j] = (p_ij / (p_i * p_j)).ln().max(0.0);
                    } else {
                        cooccur[i * vocab_size + j] = 0.0;
                    }
                }
            }
        }

        let embeddings = truncated_svd(&cooccur, vocab_size, dim, 20, &mut rng);
        let transform = build_transform(&embeddings, dim, &mut rng);

        Self {
            dim,
            vocab_size,
            embeddings,
            transform,
        }
    }

    /// Convert a token ID to a Wave in the semantic phase space.
    pub fn token_to_wave(&self, token_id: usize) -> Wave {
        if token_id >= self.vocab_size {
            // Unknown token → uniform wave
            let data: Vec<Complex64> = (0..self.dim)
                .map(|_| Complex64::new(1.0 / (self.dim as f64).sqrt(), 0.0))
                .collect();
            return Wave {
                amplitudes: ComplexVec::new(data).normalized(),
                t: 0.0,
                density: None,
                dim: self.dim,
                metrics: CollapseMetrics::new(),
            };
        }

        let emb = &self.embeddings[token_id];

        // Convert embedding to complex amplitudes:
        // magnitude = |embedding value|, phase = sign-based
        let data: Vec<Complex64> = emb.iter().enumerate()
            .map(|(_i, &val)| {
                let mag = (0.1 + 0.9 * val.abs()).max(1e-8); // full range preserved, no zero
                let phase = val * std::f64::consts::PI; // embedding value directly encodes phase (continuous)
                Complex64::from_polar(mag, phase)
            })
            .collect();

        Wave {
            amplitudes: ComplexVec::new(data).normalized(),
            t: 0.0,
            density: None,
            dim: self.dim,
            metrics: CollapseMetrics::new(),
        }
    }

    /// Convert a sequence of token IDs to waves.
    pub fn tokens_to_waves(&self, token_ids: &[usize]) -> Vec<Wave> {
        token_ids.iter().map(|&id| self.token_to_wave(id)).collect()
    }

    /// Update an embedding using gradient descent (Phase 3).
    ///
    /// Propagates gradients from the observatory back to the embedding layer
    /// for end-to-end training.
    pub fn update_embedding(&mut self, token_id: usize, gradient: &[f64], lr: f64) {
        if token_id >= self.vocab_size {
            return;
        }
        let emb = &mut self.embeddings[token_id];
        for (i, val) in emb.iter_mut().enumerate() {
            if i < gradient.len() {
                *val -= lr * gradient[i];
            }
        }
        // Re-normalize
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in emb.iter_mut() {
                *val /= norm;
            }
        }
    }

    /// Convert a token ID directly to a complex embedding vector.
    ///
    /// Unlike `token_to_wave()` which normalizes and wraps in a Wave struct,
    /// this returns the raw complex embedding for use in reservoir processing.
    /// Each embedding dimension maps to a Complex64:
    ///   magnitude = |val|^0.5, phase = sign-based + index rotation.
    pub fn token_to_complex_embedding(&self, token_id: usize) -> Vec<Complex64> {
        if token_id >= self.vocab_size {
            return (0..self.dim)
                .map(|_| Complex64::new(1.0 / (self.dim as f64).sqrt(), 0.0))
                .collect();
        }
        let emb = &self.embeddings[token_id];
        emb.iter().enumerate()
            .map(|(_i, &val)| {
                let mag = (0.1 + 0.9 * val.abs()).max(1e-8);
                let phase = val * std::f64::consts::PI;
                Complex64::from_polar(mag, phase)
            })
            .collect()
    }

    /// Cosine similarity between two token embeddings.
    pub fn similarity(&self, id_a: usize, id_b: usize) -> f64 {
        if id_a >= self.vocab_size || id_b >= self.vocab_size {
            return 0.0;
        }
        let a = &self.embeddings[id_a];
        let b = &self.embeddings[id_b];
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// SVD via power iteration (simple, no external deps)
// ---------------------------------------------------------------------------

/// Truncated SVD using randomized power iteration.
/// Returns embeddings matrix (n_rows × target_dim).
fn truncated_svd(
    matrix: &[f64],
    n: usize,
    target_dim: usize,
    n_iter: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<f64>> {
    let dim = target_dim.min(n);

    // Random initial matrix (n × dim)
    let mut q: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..n).map(|_| rng.gen::<f64>() - 0.5).collect())
        .collect();

    // Power iteration: Q = orth(M @ M^T @ Q) repeatedly
    for _ in 0..n_iter {
        // Y = M @ Q^T columns → for each column of Q, compute M @ col
        let mut y: Vec<Vec<f64>> = vec![vec![0.0; n]; dim];
        for k in 0..dim {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += matrix[i * n + j] * q[k][j];
                }
                y[k][i] = sum;
            }
        }
        // Z = M^T @ Y → M @ Y (symmetric)
        let mut z: Vec<Vec<f64>> = vec![vec![0.0; n]; dim];
        for k in 0..dim {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += matrix[i * n + j] * y[k][j];
                }
                z[k][i] = sum;
            }
        }
        // Modified Gram-Schmidt orthogonalization
        q = gram_schmidt(&z, n);
    }

    // Extract embeddings: project each row of matrix onto Q columns
    let mut embeddings: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
    for i in 0..n {
        for k in 0..dim {
            let mut dot = 0.0;
            for j in 0..n {
                dot += matrix[i * n + j] * q[k][j];
            }
            embeddings[i][k] = dot;
        }
    }

    // Normalize each embedding
    for emb in embeddings.iter_mut() {
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in emb.iter_mut() {
                *val /= norm;
            }
        }
    }

    embeddings
}

/// Modified Gram-Schmidt orthogonalization.
fn gram_schmidt(vectors: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let k = vectors.len();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(k);

    for i in 0..k {
        let mut v = vectors[i].clone();

        // Subtract projections onto previous orthogonal vectors
        for j in 0..q.len() {
            let dot: f64 = v.iter().zip(q[j].iter()).map(|(a, b)| a * b).sum();
            for idx in 0..n {
                v[idx] -= dot * q[j][idx];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in v.iter_mut() {
                *val /= norm;
            }
            q.push(v);
        } else {
            // Degenerate: push a small random vector
            q.push(vec![1e-6; n]);
        }
    }

    q
}

/// Build a transformation matrix from embedding structure.
/// Uses correlation between embeddings to create a meaningful transform.
fn build_transform(embeddings: &[Vec<f64>], dim: usize, rng: &mut ChaCha8Rng) -> TransMatrix {
    let n = embeddings.len();
    let mut data = vec![0.0f32; dim * dim];

    // Initialize with correlation structure from embeddings
    // M[i][j] = average correlation between dim i and dim j across all tokens
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0;
            let mut count = 0;
            for emb in embeddings.iter().take(n) {
                if i < emb.len() && j < emb.len() {
                    sum += emb[i] * emb[j];
                    count += 1;
                }
            }
            if count > 0 {
                data[i * dim + j] = (sum / count as f64) as f32;
            }
        }
    }

    // Add identity for stability + small random perturbation
    for i in 0..dim {
        data[i * dim + i] += 0.5;
        for j in 0..dim {
            data[i * dim + j] += (rng.gen::<f32>() - 0.5) * 0.01;
        }
    }

    // Normalize rows (spectral radius control)
    for i in 0..dim {
        let row_norm: f32 = (0..dim)
            .map(|j| data[i * dim + j] * data[i * dim + j])
            .sum::<f32>()
            .sqrt();
        if row_norm > 1e-6 {
            for j in 0..dim {
                data[i * dim + j] /= row_norm;
            }
        }
    }

    TransMatrix::new(data, dim, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tokenizer::Vocabulary;

    #[test]
    fn test_sps_construction() {
        let corpus = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "a cat and a dog",
        ];
        let vocab = Vocabulary::from_corpus(&corpus, 100);
        let sps = SemanticPhaseSpace::from_corpus(&vocab, &corpus, 16, 42);

        assert_eq!(sps.dim, 16);
        assert_eq!(sps.vocab_size, vocab.size);
    }

    #[test]
    fn test_token_to_wave() {
        let corpus = vec!["hello world"];
        let vocab = Vocabulary::from_corpus(&corpus, 100);
        let sps = SemanticPhaseSpace::from_corpus(&vocab, &corpus, 8, 42);

        let wave = sps.token_to_wave(vocab.encode_word("hello"));
        assert_eq!(wave.dim, 8);
        assert!(!wave.is_collapsed());

        // Probabilities should sum to ~1.0
        let probs = wave.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs sum = {}", sum);
    }

    #[test]
    fn test_similarity() {
        let corpus = vec![
            "the cat sat on the mat",
            "the cat ate the fish",
            "the dog sat on the log",
            "the dog ate the bone",
        ];
        let vocab = Vocabulary::from_corpus(&corpus, 100);
        let sps = SemanticPhaseSpace::from_corpus(&vocab, &corpus, 16, 42);

        // "cat" and "dog" should have some similarity (both animals, similar context)
        let cat_id = vocab.encode_word("cat");
        let dog_id = vocab.encode_word("dog");
        let mat_id = vocab.encode_word("mat");
        let sim_cat_dog = sps.similarity(cat_id, dog_id);
        let sim_cat_mat = sps.similarity(cat_id, mat_id);
        // cat-dog should have higher similarity than cat-mat in context
        // (both appear with "sat on the", "ate the")
        println!("sim(cat,dog) = {:.4}, sim(cat,mat) = {:.4}", sim_cat_dog, sim_cat_mat);
    }
}
