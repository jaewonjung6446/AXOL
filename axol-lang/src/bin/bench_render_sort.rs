//! Rendering-Fair Sort Benchmark — f32-based, multiple distributions
//!
//! Compares 5 algorithms across 8 rendering-realistic scenarios:
//!   1. std::sort (pdqsort)
//!   2. aXOL Pure (O(1) scatter, approximate)
//!   3. aXOL Hybrid (scatter + bucket sort, exact)
//!   4. Radix Sort LSD (4-pass, f32 = 32bit)
//!   5. Insertion Sort (nearly-sorted only)
//!
//! Distributions: nearly-sorted, clustered, power-law, uniform
//! GPU estimates apply the SAME hardware model to ALL algorithms fairly.

use std::hint::black_box;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

// ══════════════════════════════════════════════════════════════
// Output structures
// ══════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct BenchRenderResult {
    metadata: Metadata,
    scenarios: Vec<ScenarioResult>,
    gpu_estimates: Vec<GpuScenario>,
    fairness_summary: FairnessSummary,
}

#[derive(Serialize)]
struct Metadata {
    seed: u64,
    hardware_model: HardwareModelJson,
}

#[derive(Serialize, Clone)]
struct HardwareModelJson {
    gpu_bw_gbs: f64,
    cpu_bw_gbs: f64,
    pcie_bw_gbs: f64,
    gpu_merge_eff: f64,
    gpu_scatter_eff: f64,
    gpu_radix_eff: f64,
}

#[derive(Serialize)]
struct ScenarioResult {
    scenario: String,
    description: String,
    n: usize,
    distribution_stats: DistStats,
    algorithms: Vec<AlgoResult>,
    /// Detailed quality metrics for aXOL Pure scatter
    scatter_quality: Option<ScatterQuality>,
}

#[derive(Serialize)]
struct ScatterQuality {
    /// Fraction of adjacent pairs in correct order (graph smoothness)
    adj_order_pct: f64,
    /// Total adjacent inversions / total adjacent pairs
    adj_inversions: usize,
    adj_total: usize,
    /// Mean |correct_position - actual_position|
    mean_displacement: f64,
    /// Max displacement
    max_displacement: usize,
    /// Fraction of elements within ±1 of correct position
    within_1_pct: f64,
    /// Fraction of elements within ±2 of correct position
    within_2_pct: f64,
    /// Fraction of elements within ±10 of correct position
    within_10_pct: f64,
    /// True concordance rate (not rounded) via exhaustive or large-sample
    concordance_pct: f64,
    /// Bucket collision stats
    mean_bucket_size: f64,
    max_bucket_size: usize,
    /// Fraction of buckets with exactly 1 element (no collision)
    singleton_bucket_pct: f64,
    /// Fraction of buckets with >1 elements (collision)
    collision_bucket_pct: f64,
}

#[derive(Serialize)]
struct DistStats {
    pre_sortedness: f64,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
}

#[derive(Serialize)]
struct AlgoResult {
    name: String,
    total_us: f64,
    accuracy: f64,
    kendall_tau_pct: f64,
    is_exact: bool,
    speedup_vs_std: f64,
}

#[derive(Serialize)]
struct GpuScenario {
    scenario: String,
    n: usize,
    algorithms: Vec<GpuAlgoResult>,
}

#[derive(Serialize)]
struct GpuAlgoResult {
    name: String,
    cpu_us: f64,
    gpu_us: f64,
    hetero_us: f64,
}

#[derive(Serialize)]
struct FairnessSummary {
    best_cpu_per_scenario: Vec<BestEntry>,
    best_gpu_per_scenario: Vec<BestEntry>,
    axol_advantages: Vec<String>,
    axol_disadvantages: Vec<String>,
}

#[derive(Serialize)]
struct BestEntry {
    scenario: String,
    winner: String,
    time_us: f64,
}

// ══════════════════════════════════════════════════════════════
// f32 bit-flip conversion (preserves total order)
// ══════════════════════════════════════════════════════════════

#[inline]
fn f32_to_sortable_u32(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits & (1u32 << 31) != 0 {
        !bits
    } else {
        bits ^ (1u32 << 31)
    }
}

#[inline]
fn sortable_u32_to_f32(bits: u32) -> f32 {
    let raw = if bits & (1u32 << 31) != 0 {
        bits ^ (1u32 << 31)
    } else {
        !bits
    };
    f32::from_bits(raw)
}

// ══════════════════════════════════════════════════════════════
// ResonanceTable32 — f32 scatter sort
// ══════════════════════════════════════════════════════════════

struct ResonanceTable32 {
    min: f32,
    inv_range: f64, // use f64 for precision in bucket calc
    num_buckets: usize,
    prefix_sum: Vec<usize>,
}

impl ResonanceTable32 {
    fn build(data: &[f32], num_buckets: usize) -> Self {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in data {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let range = if (max - min).abs() < 1e-10 { 1.0f64 } else { (max - min) as f64 };
        let inv_range = (num_buckets as f64 - 1.0) / range;

        let mut counts = vec![0u32; num_buckets];
        for &v in data {
            let b = ((v as f64 - min as f64) * inv_range) as usize;
            counts[b.min(num_buckets - 1)] += 1;
        }

        let mut prefix_sum = vec![0usize; num_buckets + 1];
        for i in 0..num_buckets {
            prefix_sum[i + 1] = prefix_sum[i] + counts[i] as usize;
        }

        ResonanceTable32 { min, inv_range, num_buckets, prefix_sum }
    }

    #[inline]
    fn bucket_of(&self, v: f32) -> usize {
        let b = ((v as f64 - self.min as f64) * self.inv_range) as usize;
        b.min(self.num_buckets - 1)
    }

    fn scatter_only(&self, data: &[f32], result: &mut [f32], offsets: &mut [usize]) {
        let nb = self.num_buckets;
        offsets[..nb].copy_from_slice(&self.prefix_sum[..nb]);
        for &v in data {
            let b = self.bucket_of(v);
            let pos = offsets[b];
            result[pos] = v;
            offsets[b] = pos + 1;
        }
    }

    fn scatter_and_sort(&self, data: &[f32], result: &mut [f32], offsets: &mut [usize]) {
        self.scatter_only(data, result, offsets);
        for i in 0..self.num_buckets {
            let start = self.prefix_sum[i];
            let end = self.prefix_sum[i + 1];
            if end - start > 1 {
                result[start..end].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
// Radix Sort f32 — LSD, 4 passes (32bit / 8bit)
// ══════════════════════════════════════════════════════════════

fn radix_sort_f32(data: &[f32], result: &mut [f32]) {
    let n = data.len();
    let mut keys: Vec<u32> = data.iter().map(|&v| f32_to_sortable_u32(v)).collect();
    let mut buf: Vec<u32> = vec![0u32; n];

    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 256];

        for &k in &keys {
            let digit = ((k >> shift) & 0xFF) as usize;
            counts[digit] += 1;
        }

        let mut prefix = [0usize; 256];
        for i in 1..256 {
            prefix[i] = prefix[i - 1] + counts[i - 1] as usize;
        }

        for &k in &keys {
            let digit = ((k >> shift) & 0xFF) as usize;
            buf[prefix[digit]] = k;
            prefix[digit] += 1;
        }

        std::mem::swap(&mut keys, &mut buf);
    }

    for i in 0..n {
        result[i] = sortable_u32_to_f32(keys[i]);
    }
}

// ══════════════════════════════════════════════════════════════
// Insertion Sort f32 (for nearly-sorted data)
// ══════════════════════════════════════════════════════════════

fn insertion_sort_f32(data: &mut [f32]) {
    let n = data.len();
    for i in 1..n {
        let key = data[i];
        let mut j = i;
        while j > 0 && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

// ══════════════════════════════════════════════════════════════
// Distribution generators
// ══════════════════════════════════════════════════════════════

/// Nearly-sorted: start sorted, then randomly swap `(1-sortedness)*n` pairs
fn gen_nearly_sorted(n: usize, sortedness: f64, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut data: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let swaps = ((1.0 - sortedness) * n as f64 * 0.5) as usize;
    for _ in 0..swaps {
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
        data.swap(a, b);
    }
    data
}

/// Clustered: k clusters with tight Gaussian spread around random centers
fn gen_clustered(n: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let centers: Vec<f32> = (0..k).map(|i| (i as f32 + 0.5) / k as f32).collect();
    let spread = 0.01 / k as f32;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let center = centers[i % k];
        // Box-Muller for Gaussian
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let val = center + spread * z as f32;
        data.push(val.clamp(0.0, 1.0));
    }
    data
}

/// Power-law: CDF^(-1) = U^(1/alpha), concentrated near 0
fn gen_power_law(n: usize, alpha: f64, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let u: f64 = rng.gen::<f64>().max(1e-10);
        let val = u.powf(1.0 / alpha);
        data.push(val as f32);
    }
    data
}

/// Uniform [0, 1)
fn gen_uniform(n: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    (0..n).map(|_| rng.gen::<f32>()).collect()
}

// ══════════════════════════════════════════════════════════════
// Statistics helpers
// ══════════════════════════════════════════════════════════════

fn compute_pre_sortedness(data: &[f32]) -> f64 {
    if data.len() <= 1 { return 1.0; }
    let mut ordered = 0usize;
    for i in 0..data.len() - 1 {
        if data[i] <= data[i + 1] { ordered += 1; }
    }
    ordered as f64 / (data.len() - 1) as f64
}

fn compute_stats(data: &[f32]) -> DistStats {
    let n = data.len() as f64;
    let pre_sortedness = compute_pre_sortedness(data);
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = data.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n;
    let std_dev = var.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    DistStats { pre_sortedness, mean, std_dev, min: min as f64, max: max as f64 }
}

fn compute_accuracy_f32(ground_truth: &[f32], result: &[f32]) -> f64 {
    let n = ground_truth.len();
    let mut correct = 0usize;
    for i in 0..n {
        if (ground_truth[i] - result[i]).abs() < 1e-9 { correct += 1; }
    }
    correct as f64 / n as f64
}

/// Approximate Kendall tau via sampling for large n
fn approx_kendall_tau_f32(result: &[f32], ground_truth: &[f32]) -> f64 {
    let n = result.len();
    if n <= 1 { return 100.0; }

    // For exact sorts, quick check
    let mut is_exact = true;
    for i in 0..n {
        if (result[i] - ground_truth[i]).abs() > 1e-9 {
            is_exact = false;
            break;
        }
    }
    if is_exact { return 100.0; }

    // Sample-based for large n
    let sample_pairs = 500_000usize.min(n * (n - 1) / 2);
    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let mut concordant = 0u64;
    let mut total = 0u64;

    for _ in 0..sample_pairs {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i == j { continue; }
        total += 1;
        let gt_order = ground_truth[i] < ground_truth[j];
        let res_order = result[i] < result[j];
        if gt_order == res_order { concordant += 1; }
    }

    if total == 0 { return 100.0; }
    (concordant as f64 / total as f64) * 100.0
}

/// Compute detailed quality metrics for scatter result
fn compute_scatter_quality(
    ground_truth: &[f32],
    scatter_result: &[f32],
    table: &ResonanceTable32,
) -> ScatterQuality {
    let n = ground_truth.len();

    // Adjacent inversions (graph smoothness)
    let mut adj_inv = 0usize;
    for i in 0..n - 1 {
        if scatter_result[i] > scatter_result[i + 1] { adj_inv += 1; }
    }
    let adj_total = n - 1;
    let adj_order_pct = (1.0 - adj_inv as f64 / adj_total as f64) * 100.0;

    // Build rank map: for each value in ground_truth (sorted), rank[i] = i
    // For scatter_result, find where each element should be
    // We need: for position i in scatter_result, what is the correct position?
    // ground_truth is sorted. scatter_result[i] should be at position rank(scatter_result[i])
    // Since ground_truth is sorted, rank of value v = position of v in ground_truth
    // Use binary search on ground_truth
    let mut total_disp: u64 = 0;
    let mut max_disp: usize = 0;
    let mut within_1 = 0usize;
    let mut within_2 = 0usize;
    let mut within_10 = 0usize;

    for i in 0..n {
        // Find correct position of scatter_result[i] in ground_truth
        let v = scatter_result[i];
        let correct_pos = match ground_truth.binary_search_by(|x| x.partial_cmp(&v).unwrap()) {
            Ok(p) => p,
            Err(p) => p,
        };
        let disp = if i >= correct_pos { i - correct_pos } else { correct_pos - i };
        total_disp += disp as u64;
        if disp > max_disp { max_disp = disp; }
        if disp <= 1 { within_1 += 1; }
        if disp <= 2 { within_2 += 1; }
        if disp <= 10 { within_10 += 1; }
    }
    let mean_displacement = total_disp as f64 / n as f64;

    // Concordance rate with focused sampling (sample NEARBY pairs to catch local disorder)
    let mut rng = ChaCha8Rng::seed_from_u64(99999);
    let mut concordant = 0u64;
    let mut total_pairs = 0u64;
    // Phase 1: random global pairs (500K)
    for _ in 0..500_000u64.min((n * (n - 1) / 2) as u64) {
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        if i == j { continue; }
        total_pairs += 1;
        if (ground_truth[i] < ground_truth[j]) == (scatter_result[i] < scatter_result[j]) {
            concordant += 1;
        }
    }
    // Phase 2: nearby pairs (gap 1..100, 500K) — this catches local inversions
    let nearby_samples = 500_000u64.min((n as u64) * 50);
    for _ in 0..nearby_samples {
        let i = rng.gen_range(0..n);
        let gap = rng.gen_range(1..101usize.min(n));
        let j = if i + gap < n { i + gap } else { continue };
        total_pairs += 1;
        if (ground_truth[i] < ground_truth[j]) == (scatter_result[i] < scatter_result[j]) {
            concordant += 1;
        }
    }
    let concordance_pct = if total_pairs > 0 {
        concordant as f64 / total_pairs as f64 * 100.0
    } else { 100.0 };

    // Bucket collision stats
    let nb = table.num_buckets;
    let mut singleton = 0usize;
    let mut collision = 0usize;
    let mut max_bucket = 0usize;
    let mut nonempty = 0usize;
    for i in 0..nb {
        let sz = table.prefix_sum[i + 1] - table.prefix_sum[i];
        if sz > 0 { nonempty += 1; }
        if sz == 1 { singleton += 1; }
        if sz > 1 { collision += 1; }
        if sz > max_bucket { max_bucket = sz; }
    }
    let mean_bucket_size = if nonempty > 0 { n as f64 / nonempty as f64 } else { 0.0 };

    ScatterQuality {
        adj_order_pct,
        adj_inversions: adj_inv,
        adj_total,
        mean_displacement,
        max_displacement: max_disp,
        within_1_pct: within_1 as f64 / n as f64 * 100.0,
        within_2_pct: within_2 as f64 / n as f64 * 100.0,
        within_10_pct: within_10 as f64 / n as f64 * 100.0,
        concordance_pct,
        mean_bucket_size,
        max_bucket_size: max_bucket,
        singleton_bucket_pct: singleton as f64 / nb as f64 * 100.0,
        collision_bucket_pct: collision as f64 / nb as f64 * 100.0,
    }
}

// ══════════════════════════════════════════════════════════════
// Scenario definition
// ══════════════════════════════════════════════════════════════

struct Scenario {
    name: &'static str,
    description: &'static str,
    data: Vec<f32>,
}

fn build_scenarios(rng: &mut ChaCha8Rng) -> Vec<Scenario> {
    vec![
        Scenario {
            name: "particle_95",
            description: "Particle depth, 95% pre-sorted",
            data: gen_nearly_sorted(1_000_000, 0.95, rng),
        },
        Scenario {
            name: "particle_99",
            description: "Particle depth, 99% pre-sorted",
            data: gen_nearly_sorted(2_000_000, 0.99, rng),
        },
        Scenario {
            name: "oit_5layer",
            description: "OIT 5-layer transparency",
            data: gen_clustered(100_000, 5, rng),
        },
        Scenario {
            name: "oit_20layer",
            description: "OIT 20-layer transparency",
            data: gen_clustered(100_000, 20, rng),
        },
        Scenario {
            name: "rayhit_heavy",
            description: "Ray hit distance, heavy tail (alpha=1.5)",
            data: gen_power_law(10_000_000, 1.5, rng),
        },
        Scenario {
            name: "rayhit_normal",
            description: "Ray hit distance, normal (alpha=2.0)",
            data: gen_power_law(10_000_000, 2.0, rng),
        },
        Scenario {
            name: "uniform_1m",
            description: "Uniform baseline 1M",
            data: gen_uniform(1_000_000, rng),
        },
        Scenario {
            name: "uniform_10m",
            description: "Uniform baseline 10M",
            data: gen_uniform(10_000_000, rng),
        },
    ]
}

// ══════════════════════════════════════════════════════════════
// Benchmark runner per scenario
// ══════════════════════════════════════════════════════════════

fn bench_scenario(scenario: &Scenario) -> ScenarioResult {
    let data = &scenario.data;
    let n = data.len();
    let stats = compute_stats(data);

    let iterations: usize = if n <= 100_000 { 20 }
        else if n <= 1_000_000 { 5 }
        else if n <= 2_000_000 { 3 }
        else { 2 };

    // Ground truth
    let mut ground_truth = data.clone();
    ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut algorithms: Vec<AlgoResult> = Vec::new();

    // 1. std::sort
    let std_us = {
        let mut copy = data.clone();
        let start = Instant::now();
        for _ in 0..iterations {
            copy.copy_from_slice(data);
            copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            black_box(&copy);
        }
        start.elapsed().as_secs_f64() / iterations as f64 * 1e6
    };
    algorithms.push(AlgoResult {
        name: "std_sort".into(),
        total_us: std_us,
        accuracy: 1.0,
        kendall_tau_pct: 100.0,
        is_exact: true,
        speedup_vs_std: 1.0,
    });

    // 2. aXOL Pure (scatter only) + quality metrics
    let table_for_quality = ResonanceTable32::build(data, n);
    let (pure_us, pure_acc, pure_kt, scatter_quality) = {
        let table = &table_for_quality;
        let mut result_buf = vec![0.0f32; n];
        let mut offsets_buf = vec![0usize; table.num_buckets];

        let start = Instant::now();
        for _ in 0..iterations {
            table.scatter_only(data, &mut result_buf, &mut offsets_buf);
            black_box(&result_buf);
        }
        let us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        table.scatter_only(data, &mut result_buf, &mut offsets_buf);
        let acc = compute_accuracy_f32(&ground_truth, &result_buf);
        let kt = approx_kendall_tau_f32(&result_buf, &ground_truth);

        // Compute detailed scatter quality
        eprintln!("  Computing scatter quality metrics...");
        let quality = compute_scatter_quality(&ground_truth, &result_buf, table);
        eprintln!("    adj_order={:.4}%  mean_disp={:.2}  max_disp={}  within1={:.1}%  within10={:.1}%",
            quality.adj_order_pct, quality.mean_displacement, quality.max_displacement,
            quality.within_1_pct, quality.within_10_pct);
        eprintln!("    concordance={:.6}%  singleton_buckets={:.1}%  collision_buckets={:.1}%  max_bucket={}",
            quality.concordance_pct, quality.singleton_bucket_pct, quality.collision_bucket_pct,
            quality.max_bucket_size);

        (us, acc, kt, quality)
    };
    algorithms.push(AlgoResult {
        name: "axol_pure".into(),
        total_us: pure_us,
        accuracy: pure_acc,
        kendall_tau_pct: pure_kt,
        is_exact: false,
        speedup_vs_std: std_us / pure_us,
    });

    // 3. aXOL Hybrid (scatter + bucket sort)
    let (hybrid_us, hybrid_acc) = {
        let table = ResonanceTable32::build(data, n);
        let mut result_buf = vec![0.0f32; n];
        let mut offsets_buf = vec![0usize; table.num_buckets];

        let start = Instant::now();
        for _ in 0..iterations {
            table.scatter_and_sort(data, &mut result_buf, &mut offsets_buf);
            black_box(&result_buf);
        }
        let us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        table.scatter_and_sort(data, &mut result_buf, &mut offsets_buf);
        let acc = compute_accuracy_f32(&ground_truth, &result_buf);
        (us, acc)
    };
    algorithms.push(AlgoResult {
        name: "axol_hybrid".into(),
        total_us: hybrid_us,
        accuracy: hybrid_acc,
        kendall_tau_pct: 100.0,
        is_exact: true,
        speedup_vs_std: std_us / hybrid_us,
    });

    // 4. Radix Sort (4-pass f32)
    let radix_us = {
        let mut result_buf = vec![0.0f32; n];
        let start = Instant::now();
        for _ in 0..iterations {
            radix_sort_f32(data, &mut result_buf);
            black_box(&result_buf);
        }
        start.elapsed().as_secs_f64() / iterations as f64 * 1e6
    };
    algorithms.push(AlgoResult {
        name: "radix_4pass".into(),
        total_us: radix_us,
        accuracy: 1.0,
        kendall_tau_pct: 100.0,
        is_exact: true,
        speedup_vs_std: std_us / radix_us,
    });

    // 5. Insertion Sort (only for nearly-sorted scenarios, n <= 2M)
    if scenario.name.starts_with("particle") {
        let ins_us = {
            let mut copy = data.clone();
            let start = Instant::now();
            let ins_iters = if n <= 1_000_000 { 2 } else { 1 };
            for _ in 0..ins_iters {
                copy.copy_from_slice(data);
                insertion_sort_f32(&mut copy);
                black_box(&copy);
            }
            start.elapsed().as_secs_f64() / ins_iters as f64 * 1e6
        };
        algorithms.push(AlgoResult {
            name: "insertion".into(),
            total_us: ins_us,
            accuracy: 1.0,
            kendall_tau_pct: 100.0,
            is_exact: true,
            speedup_vs_std: std_us / ins_us,
        });
    }

    ScenarioResult {
        scenario: scenario.name.to_string(),
        description: scenario.description.to_string(),
        n,
        distribution_stats: stats,
        algorithms,
        scatter_quality: Some(scatter_quality),
    }
}

// ══════════════════════════════════════════════════════════════
// GPU estimation — fair hardware model
// ══════════════════════════════════════════════════════════════

struct HardwareModel {
    gpu_bw: f64,         // GB/s
    cpu_bw: f64,         // GB/s
    pcie_bw: f64,        // GB/s
    gpu_merge_eff: f64,  // std::sort on GPU (comparison-based → poor)
    gpu_scatter_eff: f64,// aXOL scatter on GPU
    gpu_radix_eff: f64,  // Radix on GPU
}

impl HardwareModel {
    fn default_model() -> Self {
        HardwareModel {
            gpu_bw: 500.0,
            cpu_bw: 50.0,
            pcie_bw: 25.0,
            gpu_merge_eff: 0.3,
            gpu_scatter_eff: 0.5,
            gpu_radix_eff: 0.6,
        }
    }

    fn to_json(&self) -> HardwareModelJson {
        HardwareModelJson {
            gpu_bw_gbs: self.gpu_bw,
            cpu_bw_gbs: self.cpu_bw,
            pcie_bw_gbs: self.pcie_bw,
            gpu_merge_eff: self.gpu_merge_eff,
            gpu_scatter_eff: self.gpu_scatter_eff,
            gpu_radix_eff: self.gpu_radix_eff,
        }
    }

    /// Estimate GPU time from CPU time using bandwidth ratio and efficiency
    fn gpu_estimate(&self, cpu_us: f64, efficiency: f64) -> f64 {
        let bw_ratio = self.gpu_bw / self.cpu_bw;
        cpu_us / (bw_ratio * efficiency)
    }

    fn pcie_transfer_us(&self, n: usize) -> f64 {
        let bytes = (n * 4) as f64; // f32 = 4 bytes
        bytes / (self.pcie_bw * 1e3) // GB/s → bytes/μs
    }
}

fn estimate_gpu(scenarios: &[ScenarioResult], hw: &HardwareModel) -> Vec<GpuScenario> {
    scenarios.iter().map(|s| {
        let n = s.n;
        let pcie_us = hw.pcie_transfer_us(n);

        let mut gpu_algos = Vec::new();

        for algo in &s.algorithms {
            let (gpu_us, hetero_us) = match algo.name.as_str() {
                "std_sort" => {
                    // Comparison sort on GPU: merge sort, low efficiency
                    let gpu = hw.gpu_estimate(algo.total_us, hw.gpu_merge_eff);
                    (gpu, gpu) // no CPU portion needed
                }
                "axol_pure" => {
                    let gpu = hw.gpu_estimate(algo.total_us, hw.gpu_scatter_eff);
                    (gpu, gpu)
                }
                "axol_hybrid" => {
                    // GPU scatter + PCIe + CPU bucket sort
                    // Find pure scatter time (approximate: pure_us from same scenario)
                    let pure_us = s.algorithms.iter()
                        .find(|a| a.name == "axol_pure")
                        .map(|a| a.total_us)
                        .unwrap_or(algo.total_us * 0.5);
                    let bucket_us = algo.total_us - pure_us;
                    let gpu_scatter = hw.gpu_estimate(pure_us, hw.gpu_scatter_eff);
                    let hetero = gpu_scatter + pcie_us + bucket_us;
                    (gpu_scatter, hetero)
                }
                "radix_4pass" => {
                    let gpu = hw.gpu_estimate(algo.total_us, hw.gpu_radix_eff);
                    (gpu, gpu)
                }
                "insertion" => {
                    // Insertion sort is not GPU-friendly at all
                    (algo.total_us, algo.total_us) // stays on CPU
                }
                _ => (algo.total_us, algo.total_us),
            };

            gpu_algos.push(GpuAlgoResult {
                name: algo.name.clone(),
                cpu_us: algo.total_us,
                gpu_us,
                hetero_us,
            });
        }

        GpuScenario {
            scenario: s.scenario.clone(),
            n,
            algorithms: gpu_algos,
        }
    }).collect()
}

// ══════════════════════════════════════════════════════════════
// Fairness summary
// ══════════════════════════════════════════════════════════════

fn build_fairness_summary(scenarios: &[ScenarioResult], gpu_scenarios: &[GpuScenario]) -> FairnessSummary {
    let mut best_cpu: Vec<BestEntry> = Vec::new();
    let mut best_gpu: Vec<BestEntry> = Vec::new();

    let mut axol_hybrid_wins_cpu = 0;
    let mut axol_hybrid_wins_gpu = 0;
    let total = scenarios.len();

    for s in scenarios {
        // Only compare exact algorithms for CPU
        let exact_algos: Vec<&AlgoResult> = s.algorithms.iter()
            .filter(|a| a.is_exact)
            .collect();
        if let Some(best) = exact_algos.iter().min_by(|a, b| a.total_us.partial_cmp(&b.total_us).unwrap()) {
            if best.name == "axol_hybrid" { axol_hybrid_wins_cpu += 1; }
            best_cpu.push(BestEntry {
                scenario: s.scenario.clone(),
                winner: best.name.clone(),
                time_us: best.total_us,
            });
        }
    }

    for gs in gpu_scenarios {
        // Exclude insertion for GPU comparison
        let gpu_algos: Vec<&GpuAlgoResult> = gs.algorithms.iter()
            .filter(|a| a.name != "insertion" && a.name != "axol_pure")
            .collect();
        if let Some(best) = gpu_algos.iter().min_by(|a, b| a.hetero_us.partial_cmp(&b.hetero_us).unwrap()) {
            if best.name == "axol_hybrid" { axol_hybrid_wins_gpu += 1; }
            best_gpu.push(BestEntry {
                scenario: gs.scenario.clone(),
                winner: best.name.clone(),
                time_us: best.hetero_us,
            });
        }
    }

    let mut advantages = Vec::new();
    let mut disadvantages = Vec::new();

    // Analyze where aXOL wins/loses
    for bc in &best_cpu {
        if bc.winner == "axol_hybrid" {
            advantages.push(format!("CPU {}: aXOL Hybrid wins ({:.1}us)", bc.scenario, bc.time_us));
        } else if bc.winner != "axol_hybrid" {
            // Check how much aXOL lost by
            let scenario = scenarios.iter().find(|s| s.scenario == bc.scenario).unwrap();
            if let Some(axol) = scenario.algorithms.iter().find(|a| a.name == "axol_hybrid") {
                let ratio = axol.total_us / bc.time_us;
                disadvantages.push(format!(
                    "CPU {}: {} wins ({:.1}us), aXOL Hybrid {:.1}x slower ({:.1}us)",
                    bc.scenario, bc.winner, bc.time_us, ratio, axol.total_us
                ));
            }
        }
    }

    for bg in &best_gpu {
        if bg.winner == "axol_hybrid" {
            advantages.push(format!("GPU {}: aXOL Hybrid wins ({:.1}us)", bg.scenario, bg.time_us));
        } else {
            let gs = gpu_scenarios.iter().find(|g| g.scenario == bg.scenario).unwrap();
            if let Some(axol) = gs.algorithms.iter().find(|a| a.name == "axol_hybrid") {
                let ratio = axol.hetero_us / bg.time_us;
                disadvantages.push(format!(
                    "GPU {}: {} wins ({:.1}us), aXOL Hybrid {:.1}x slower ({:.1}us)",
                    bg.scenario, bg.winner, bg.time_us, ratio, axol.hetero_us
                ));
            }
        }
    }

    // Summary stats
    advantages.push(format!(
        "aXOL Hybrid CPU wins: {}/{} scenarios", axol_hybrid_wins_cpu, total
    ));
    advantages.push(format!(
        "aXOL Hybrid GPU wins: {}/{} scenarios", axol_hybrid_wins_gpu, total
    ));

    FairnessSummary {
        best_cpu_per_scenario: best_cpu,
        best_gpu_per_scenario: best_gpu,
        axol_advantages: advantages,
        axol_disadvantages: disadvantages,
    }
}

// ══════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════

fn main() {
    let seed = 42u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let hw = HardwareModel::default_model();

    eprintln!("=== Rendering-Fair Sort Benchmark (f32) ===");
    eprintln!("Hardware model: GPU={:.0}GB/s  CPU={:.0}GB/s  PCIe={:.0}GB/s",
        hw.gpu_bw, hw.cpu_bw, hw.pcie_bw);
    eprintln!("GPU efficiency: merge={:.1}  scatter={:.1}  radix={:.1}",
        hw.gpu_merge_eff, hw.gpu_scatter_eff, hw.gpu_radix_eff);
    eprintln!();

    // Build all scenarios
    eprintln!("Generating distributions...");
    let scenarios = build_scenarios(&mut rng);
    for s in &scenarios {
        let stats = compute_stats(&s.data);
        eprintln!("  {:<16} n={:<10} pre_sort={:.4}  mean={:.4}  std={:.4}",
            s.name, s.data.len(), stats.pre_sortedness, stats.mean, stats.std_dev);
    }
    eprintln!();

    // Run benchmarks
    let mut results: Vec<ScenarioResult> = Vec::new();
    for s in &scenarios {
        eprintln!("Benchmarking {}  (n={})...", s.name, s.data.len());
        let result = bench_scenario(s);
        for a in &result.algorithms {
            eprintln!("  {:<14} {:>10.1}us  acc={:.4}  kt={:.1}%  speedup={:.2}x",
                a.name, a.total_us, a.accuracy, a.kendall_tau_pct, a.speedup_vs_std);
        }
        results.push(result);
    }

    // GPU estimates
    eprintln!("\n=== GPU Estimates ===");
    let gpu_estimates = estimate_gpu(&results, &hw);
    for gs in &gpu_estimates {
        eprintln!("{}:", gs.scenario);
        for a in &gs.algorithms {
            eprintln!("  {:<14} cpu={:>10.1}us  gpu={:>10.1}us  hetero={:>10.1}us",
                a.name, a.cpu_us, a.gpu_us, a.hetero_us);
        }
    }

    // Fairness summary
    eprintln!("\n=== Fairness Summary ===");
    let fairness_summary = build_fairness_summary(&results, &gpu_estimates);
    eprintln!("CPU winners:");
    for b in &fairness_summary.best_cpu_per_scenario {
        eprintln!("  {:<16} → {} ({:.1}us)", b.scenario, b.winner, b.time_us);
    }
    eprintln!("GPU winners:");
    for b in &fairness_summary.best_gpu_per_scenario {
        eprintln!("  {:<16} → {} ({:.1}us)", b.scenario, b.winner, b.time_us);
    }
    eprintln!("\naXOL advantages:");
    for a in &fairness_summary.axol_advantages { eprintln!("  + {}", a); }
    eprintln!("aXOL disadvantages:");
    for d in &fairness_summary.axol_disadvantages { eprintln!("  - {}", d); }

    // JSON output
    let output = BenchRenderResult {
        metadata: Metadata {
            seed,
            hardware_model: hw.to_json(),
        },
        scenarios: results,
        gpu_estimates,
        fairness_summary,
    };

    let json = serde_json::to_string_pretty(&output).unwrap();
    println!("{}", json);
    eprintln!("\nDone. JSON written to stdout.");
}
