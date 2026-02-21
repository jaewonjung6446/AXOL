//! Resonance Sort Benchmark — Five-way comparison
//!
//! 1. std::sort           — O(n log n), 100% accurate
//! 2. Pure O(1) scatter   — O(1)/elem, ~63% accuracy
//! 3. Multi-slit scatter  — O(k)/elem (k slits), accuracy scales with k
//! 4. Hybrid              — O(1)/elem scatter + bucket sort, 100% accurate
//! 5. Radix Sort (LSD)    — O(n·d), 100% accurate, non-comparison
//!
//! Multi-slit: like a quantum multi-slit experiment —
//!   k different quantizations vote on each element's position.
//!   Each "slit" uses a different bucket count (n, n+Δ, n+2Δ, ...).
//!   Elements that collide in one slit may separate in another.
//!   Median position across k slits → higher accuracy.

use std::hint::black_box;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

// ── Output structures ──

#[derive(Serialize)]
struct BenchResult {
    benchmarks: Vec<BenchEntry>,
    heterogeneous: Vec<HeteroEntry>,
    ensemble_sweep: Vec<EnsembleEntry>,
    multi_slit_sweep: Vec<SlitEntry>,
    visualization: VisData,
    large_vis: Option<LargeVisData>,
}

#[derive(Serialize)]
struct BenchEntry {
    n: usize,
    std_sort_us: f64,
    std_ns_per_elem: f64,
    pure_sort_us: f64,
    pure_ns_per_elem: f64,
    pure_accuracy: f64,
    pure_speedup: f64,
    hybrid_sort_us: f64,
    hybrid_ns_per_elem: f64,
    hybrid_accuracy: f64,
    hybrid_speedup: f64,
    radix_sort_us: f64,
    radix_ns_per_elem: f64,
    radix_speedup: f64,
    // Phase breakdown (CPU measured)
    scatter_phase_us: f64,
    bucket_sort_phase_us: f64,
    resonance_build_us: f64,
}

/// GPU+CPU heterogeneous estimate
#[derive(Serialize)]
struct HeteroEntry {
    n: usize,
    /// CPU-measured scatter time
    cpu_scatter_us: f64,
    /// CPU-measured bucket-sort time
    cpu_bucket_sort_us: f64,
    /// Estimated GPU scatter time (bandwidth-proportional)
    gpu_scatter_us: f64,
    /// PCIe transfer estimate (n * 8 bytes / bandwidth)
    pcie_transfer_us: f64,
    /// GPU(scatter) + transfer + CPU(bucket sort) total
    hetero_total_us: f64,
    /// CPU-only hybrid total for comparison
    cpu_hybrid_us: f64,
    /// std::sort for comparison
    std_sort_us: f64,
    /// Radix sort for comparison
    radix_sort_us: f64,
    /// GPU radix estimate (8 passes, bandwidth-proportional)
    gpu_radix_us: f64,
    /// Speedup: std / hetero
    hetero_vs_std: f64,
    /// Speedup: hetero vs cpu_hybrid
    hetero_vs_cpu_hybrid: f64,
    /// Speedup: hetero vs gpu_radix
    hetero_vs_gpu_radix: f64,
    /// GPU bandwidth assumption (GB/s)
    gpu_bandwidth_gbs: f64,
    /// CPU bandwidth assumption (GB/s)
    cpu_bandwidth_gbs: f64,
    /// PCIe bandwidth assumption (GB/s)
    pcie_bandwidth_gbs: f64,
}

#[derive(Serialize)]
struct SlitEntry {
    k: usize,
    n: usize,
    accuracy: f64,
    sort_us: f64,
    ns_per_elem: f64,
    speedup_vs_std: f64,
}

#[derive(Serialize)]
struct VisData {
    n: usize,
    ground_truth: Vec<f64>,
    pure_result: Vec<f64>,
    pure_error_indices: Vec<usize>,
    hybrid_result: Vec<f64>,
    hybrid_error_indices: Vec<usize>,
}

/// Pre-computed stats + downsampled arrays for large-n visualization
#[derive(Serialize)]
struct LargeVisData {
    n: usize,
    exact_match_pct: f64,
    /// Total inversions (all pairs) for Kendall tau
    total_inversions: u64,
    total_pairs: u64,
    kendall_tau_pct: f64,
    /// Adjacent inversions (consecutive pairs only)
    adj_inversions: usize,
    adj_total: usize,
    adj_order_pct: f64,
    mean_displacement: f64,
    max_displacement: usize,
    within_2_pct: f64,
    gt_downsampled: Vec<f64>,
    scatter_downsampled: Vec<f64>,
    zoom_gt: Vec<f64>,
    zoom_scatter: Vec<f64>,
    zoom_start_index: usize,
}

// ── Resonance Sort ──

struct ResonanceTable {
    min: f64,
    inv_range: f64,
    num_buckets: usize,
    prefix_sum: Vec<usize>,
}

impl ResonanceTable {
    fn build(data: &[f64], num_buckets: usize) -> Self {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in data {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let inv_range = (num_buckets as f64 - 1.0) / range;

        let mut counts = vec![0u32; num_buckets];
        for &v in data {
            let b = ((v - min) * inv_range) as usize;
            counts[b.min(num_buckets - 1)] += 1;
        }

        let mut prefix_sum = vec![0usize; num_buckets + 1];
        for i in 0..num_buckets {
            prefix_sum[i + 1] = prefix_sum[i] + counts[i] as usize;
        }

        ResonanceTable { min, inv_range, num_buckets, prefix_sum }
    }

    #[inline]
    fn bucket_of(&self, v: f64) -> usize {
        let b = ((v - self.min) * self.inv_range) as usize;
        b.min(self.num_buckets - 1)
    }

    /// Pure O(1) scatter
    fn scatter_only(&self, data: &[f64], result: &mut [f64], offsets: &mut [usize]) {
        let nb = self.num_buckets;
        offsets[..nb].copy_from_slice(&self.prefix_sum[..nb]);
        for &v in data {
            let b = self.bucket_of(v);
            let pos = unsafe { *offsets.get_unchecked(b) };
            unsafe { *result.get_unchecked_mut(pos) = v; }
            unsafe { *offsets.get_unchecked_mut(b) = pos + 1; }
        }
    }

    /// Returns the position assigned to each input element (by input index)
    fn scatter_positions(&self, data: &[f64]) -> Vec<usize> {
        let n = data.len();
        let nb = self.num_buckets;
        let mut offsets: Vec<usize> = self.prefix_sum[..nb].to_vec();
        let mut positions = vec![0usize; n];
        for (i, &v) in data.iter().enumerate() {
            let b = self.bucket_of(v);
            positions[i] = offsets[b];
            offsets[b] += 1;
        }
        positions
    }

    /// Hybrid: scatter + within-bucket sort
    fn scatter_and_sort(&self, data: &[f64], result: &mut [f64], offsets: &mut [usize]) {
        self.scatter_only(data, result, offsets);
        for i in 0..self.num_buckets {
            let start = self.prefix_sum[i];
            let end = self.prefix_sum[i + 1];
            if end - start > 1 {
                result[start..end].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            }
        }
    }

    fn sort_pure(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0f64; n];
        let mut offsets = vec![0usize; self.num_buckets];
        self.scatter_only(data, &mut result, &mut offsets);
        result
    }

    fn sort_hybrid(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0f64; n];
        let mut offsets = vec![0usize; self.num_buckets];
        self.scatter_and_sort(data, &mut result, &mut offsets);
        result
    }
}

// ── Multi-slit scatter ──
// Each slit offsets the bucket boundaries by a fraction of the bucket width.
// Slit i uses offset = i/k, so boundaries shift like a multi-slit diffraction pattern.
// Elements colliding at a boundary in one slit are separated in another.

struct OffsetTable {
    min: f64,
    inv_range: f64,
    offset: f64, // [0, 1) fractional bucket offset
    num_buckets: usize,
    prefix_sum: Vec<usize>,
}

impl OffsetTable {
    fn build(data: &[f64], num_buckets: usize, offset: f64) -> Self {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in data { if v < min { min = v; } if v > max { max = v; } }
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let inv_range = (num_buckets as f64) / range;

        let mut counts = vec![0u32; num_buckets];
        for &v in data {
            let raw = (v - min) * inv_range + offset;
            let b = (raw as usize).min(num_buckets - 1);
            counts[b] += 1;
        }

        let mut prefix_sum = vec![0usize; num_buckets + 1];
        for i in 0..num_buckets {
            prefix_sum[i + 1] = prefix_sum[i] + counts[i] as usize;
        }
        OffsetTable { min, inv_range, offset, num_buckets, prefix_sum }
    }

    #[inline]
    fn bucket_of(&self, v: f64) -> usize {
        let raw = (v - self.min) * self.inv_range + self.offset;
        (raw as usize).min(self.num_buckets - 1)
    }

    /// Returns position assigned to each input element
    fn scatter_positions(&self, data: &[f64]) -> Vec<usize> {
        let n = data.len();
        let nb = self.num_buckets;
        let mut offsets: Vec<usize> = self.prefix_sum[..nb].to_vec();
        let mut positions = vec![0usize; n];
        for (i, &v) in data.iter().enumerate() {
            let b = self.bucket_of(v);
            positions[i] = offsets[b];
            offsets[b] += 1;
        }
        positions
    }
}

// ── Radix Sort for f64 (LSD, 8-bit digits, 8 passes) ──

/// Convert f64 to u64 preserving total order.
/// Positive f64: flip MSB only (0→1). Negative f64: flip all bits.
#[inline]
fn f64_to_sortable_u64(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits & (1u64 << 63) != 0 {
        !bits // negative: flip all
    } else {
        bits ^ (1u64 << 63) // positive: flip MSB
    }
}

#[inline]
fn sortable_u64_to_f64(bits: u64) -> f64 {
    let raw = if bits & (1u64 << 63) != 0 {
        bits ^ (1u64 << 63) // was positive
    } else {
        !bits // was negative
    };
    f64::from_bits(raw)
}

/// LSD Radix Sort — 8 passes × 8-bit digit = 64 bits
fn radix_sort_f64(data: &[f64], result: &mut [f64]) {
    let n = data.len();
    let mut keys: Vec<u64> = data.iter().map(|&v| f64_to_sortable_u64(v)).collect();
    let mut buf: Vec<u64> = vec![0u64; n];

    for pass in 0..8u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 256];

        // Count
        for &k in &keys {
            let digit = ((k >> shift) & 0xFF) as usize;
            counts[digit] += 1;
        }

        // Prefix sum
        let mut prefix = [0usize; 256];
        for i in 1..256 {
            prefix[i] = prefix[i - 1] + counts[i - 1] as usize;
        }

        // Scatter
        for &k in &keys {
            let digit = ((k >> shift) & 0xFF) as usize;
            buf[prefix[digit]] = k;
            prefix[digit] += 1;
        }

        std::mem::swap(&mut keys, &mut buf);
    }

    // Convert back to f64
    for i in 0..n {
        result[i] = sortable_u64_to_f64(keys[i]);
    }
}

/// k slits with offset boundaries → median position → sort by median (tie-break by value)
fn multi_slit_sort(data: &[f64], k: usize) -> Vec<f64> {
    let n = data.len();

    // Collect k position votes per element
    let mut pos_sum: Vec<f64> = vec![0.0; n];

    for slit in 0..k {
        let offset = slit as f64 / k as f64; // 0/k, 1/k, 2/k, ...
        let table = OffsetTable::build(data, n, offset);
        let positions = table.scatter_positions(data);
        for i in 0..n {
            pos_sum[i] += positions[i] as f64;
        }
    }

    // Average position per element
    let mut indexed: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        indexed.push((pos_sum[i] / k as f64, i));
    }

    // Sort by average position, tie-break by value
    indexed.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap().then(
            data[a.1].partial_cmp(&data[b.1]).unwrap()
        )
    });

    let mut result = vec![0.0f64; n];
    for (out_pos, &(_, orig_idx)) in indexed.iter().enumerate() {
        result[out_pos] = data[orig_idx];
    }
    result
}

// ── Ensemble Voting Sort ──
// k개의 서로 다른 양자화를 병렬 실행 → 위치가 일치하는 원소는 "확정" → 나머지만 정렬
//
// 1. k scatters (다른 offset) → k개의 결과 배열
// 2. 모든 k에서 같은 값이 같은 위치에 있으면 "확정"
// 3. 미확정 원소만 모아서 정렬 후 빈 자리에 삽입

#[derive(Serialize)]
struct EnsembleEntry {
    k: usize,
    n: usize,
    confirmed_pct: f64,      // 확정된 원소 비율
    unconfirmed_count: usize, // 미확정 원소 수
    accuracy: f64,
    sort_us: f64,
    ns_per_elem: f64,
    speedup_vs_std: f64,
    speedup_vs_hybrid: f64,
}

fn ensemble_voting_sort(data: &[f64], k: usize) -> (Vec<f64>, usize) {
    let n = data.len();

    // Step 1: Run k scatters with different offsets
    let mut scatter_results: Vec<Vec<f64>> = Vec::with_capacity(k);
    for slit in 0..k {
        let offset = slit as f64 / k as f64;
        let table = OffsetTable::build(data, n, offset);
        let nb = table.num_buckets;
        let mut result = vec![0.0f64; n];
        let mut offsets: Vec<usize> = table.prefix_sum[..nb].to_vec();
        for &v in data {
            let b = table.bucket_of(v);
            let pos = offsets[b];
            result[pos] = v;
            offsets[b] += 1;
        }
        scatter_results.push(result);
    }

    // Step 2: Unanimous — all k scatters agree at position i
    let mut final_result = vec![0.0f64; n];
    let mut confirmed = vec![false; n];
    let mut confirmed_count = 0usize;

    for i in 0..n {
        let val = scatter_results[0][i];
        let mut all_agree = true;
        for s in 1..k {
            if (scatter_results[s][i] - val).abs() > 1e-12 {
                all_agree = false;
                break;
            }
        }
        if all_agree {
            final_result[i] = val;
            confirmed[i] = true;
            confirmed_count += 1;
        }
    }

    // Step 3: Collect unconfirmed values
    let mut all_vals: Vec<f64> = data.to_vec();
    all_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut confirmed_vals: Vec<f64> = (0..n)
        .filter(|&i| confirmed[i])
        .map(|i| final_result[i])
        .collect();
    confirmed_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut unconfirmed_values: Vec<f64> = Vec::new();
    let mut ci = 0;
    for &v in &all_vals {
        if ci < confirmed_vals.len() && (confirmed_vals[ci] - v).abs() < 1e-12 {
            ci += 1;
        } else {
            unconfirmed_values.push(v);
        }
    }

    let mut unconfirmed_positions: Vec<usize> = Vec::new();
    for i in 0..n {
        if !confirmed[i] {
            unconfirmed_positions.push(i);
        }
    }

    // Step 4: Sort unconfirmed and fill gaps
    unconfirmed_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    unconfirmed_positions.sort_unstable();

    for (idx, &pos) in unconfirmed_positions.iter().enumerate() {
        if idx < unconfirmed_values.len() {
            final_result[pos] = unconfirmed_values[idx];
        }
    }

    (final_result, confirmed_count)
}

fn compute_accuracy(ground_truth: &[f64], result: &[f64]) -> (f64, Vec<usize>) {
    let n = ground_truth.len();
    let mut correct = 0usize;
    let mut errors = Vec::new();
    for i in 0..n {
        if (ground_truth[i] - result[i]).abs() < 1e-12 {
            correct += 1;
        } else {
            errors.push(i);
        }
    }
    (correct as f64 / n as f64, errors)
}

// ── Benchmark runners ──

fn bench_one(n: usize, rng: &mut ChaCha8Rng) -> BenchEntry {
    let iterations: usize = if n <= 10_000 { 1000 }
        else if n <= 100_000 { 100 }
        else if n <= 1_000_000 { 10 }
        else if n <= 10_000_000 { 3 }
        else { 1 };
    let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 1_000_000.0).collect();

    // std::sort
    let mut copy = data.clone();
    let start = Instant::now();
    for _ in 0..iterations {
        copy.copy_from_slice(&data);
        copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        black_box(&copy);
    }
    let std_sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;
    drop(copy); // free 800MB for large n

    // Build
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(ResonanceTable::build(&data, n));
    }
    let resonance_build_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

    let table = ResonanceTable::build(&data, n);
    let mut result_buf = vec![0.0f64; n];
    let mut offsets_buf = vec![0usize; n];

    // Pure O(1) scatter
    let start = Instant::now();
    for _ in 0..iterations {
        table.scatter_only(&data, &mut result_buf, &mut offsets_buf);
        black_box(&result_buf);
    }
    let pure_sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

    let mut ground_truth = data.clone();
    ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    table.scatter_only(&data, &mut result_buf, &mut offsets_buf);
    let (pure_accuracy, _) = compute_accuracy(&ground_truth, &result_buf);

    // Hybrid (total)
    let start = Instant::now();
    for _ in 0..iterations {
        table.scatter_and_sort(&data, &mut result_buf, &mut offsets_buf);
        black_box(&result_buf);
    }
    let hybrid_sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;
    table.scatter_and_sort(&data, &mut result_buf, &mut offsets_buf);
    let (hybrid_accuracy, _) = compute_accuracy(&ground_truth, &result_buf);

    // Phase breakdown
    let scatter_phase_us = pure_sort_us;
    let bucket_sort_phase_us = hybrid_sort_us - pure_sort_us;

    // Free resonance buffers before radix sort (saves ~3.2GB at n=100M)
    drop(table);
    drop(result_buf);
    drop(offsets_buf);
    drop(ground_truth);

    // Radix Sort
    let mut radix_buf = vec![0.0f64; n];
    let start = Instant::now();
    for _ in 0..iterations {
        radix_sort_f64(&data, &mut radix_buf);
        black_box(&radix_buf);
    }
    let radix_sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

    let nf = n as f64;
    BenchEntry {
        n,
        std_sort_us,
        std_ns_per_elem: std_sort_us * 1000.0 / nf,
        pure_sort_us,
        pure_ns_per_elem: pure_sort_us * 1000.0 / nf,
        pure_accuracy,
        pure_speedup: std_sort_us / pure_sort_us,
        hybrid_sort_us,
        hybrid_ns_per_elem: hybrid_sort_us * 1000.0 / nf,
        hybrid_accuracy,
        hybrid_speedup: std_sort_us / hybrid_sort_us,
        radix_sort_us,
        radix_ns_per_elem: radix_sort_us * 1000.0 / nf,
        radix_speedup: std_sort_us / radix_sort_us,
        scatter_phase_us,
        bucket_sort_phase_us,
        resonance_build_us,
    }
}

/// Estimate GPU+CPU heterogeneous performance from CPU measurements.
///
/// Assumptions (conservative, consumer-grade):
///   GPU memory bandwidth: 500 GB/s (RTX 3060~4070 class)
///   CPU memory bandwidth:  50 GB/s (DDR4/DDR5 dual-channel)
///   PCIe 4.0 x16:          25 GB/s (one direction)
///   Unified memory:          0 GB/s transfer (Apple M, AMD APU)
fn bench_heterogeneous(entries: &[BenchEntry]) -> Vec<HeteroEntry> {
    let gpu_bw: f64 = 500.0;   // GB/s
    let cpu_bw: f64 = 50.0;    // GB/s
    let pcie_bw: f64 = 25.0;   // GB/s

    // GPU scatter speedup from bandwidth ratio.
    // Scatter is memory-bound (random writes), so speedup ≈ bandwidth ratio.
    // But GPU atomic contention for histogram reduces effective speedup.
    // Conservative: use bandwidth_ratio * 0.5 for scatter.
    let scatter_gpu_factor = (gpu_bw / cpu_bw) * 0.5; // ~5x

    // GPU radix sort: 8 sequential passes but each pass is bandwidth-bound.
    // GPU radix implementations achieve ~80% of peak bandwidth.
    // Factor = bandwidth_ratio * 0.8 / 1.0 (CPU radix already ~100% bandwidth)
    // But CPU radix has 8 passes too, so factor is just bandwidth ratio.
    let radix_gpu_factor = (gpu_bw / cpu_bw) * 0.6; // ~6x (conservative)

    entries.iter().map(|b| {
        let n = b.n;
        let data_bytes = (n * 8) as f64; // f64 = 8 bytes

        // GPU scatter time estimate
        let gpu_scatter_us = b.scatter_phase_us / scatter_gpu_factor;

        // PCIe transfer: send result array back to CPU
        let pcie_transfer_us = data_bytes / (pcie_bw * 1e3); // GB/s → bytes/μs

        // Heterogeneous total: GPU scatter + PCIe transfer + CPU bucket sort
        let hetero_total_us = gpu_scatter_us + pcie_transfer_us + b.bucket_sort_phase_us;

        // GPU radix estimate
        let gpu_radix_us = b.radix_sort_us / radix_gpu_factor;

        HeteroEntry {
            n,
            cpu_scatter_us: b.scatter_phase_us,
            cpu_bucket_sort_us: b.bucket_sort_phase_us,
            gpu_scatter_us,
            pcie_transfer_us,
            hetero_total_us,
            cpu_hybrid_us: b.hybrid_sort_us,
            std_sort_us: b.std_sort_us,
            radix_sort_us: b.radix_sort_us,
            gpu_radix_us,
            hetero_vs_std: b.std_sort_us / hetero_total_us,
            hetero_vs_cpu_hybrid: b.hybrid_sort_us / hetero_total_us,
            hetero_vs_gpu_radix: gpu_radix_us / hetero_total_us,
            gpu_bandwidth_gbs: gpu_bw,
            cpu_bandwidth_gbs: cpu_bw,
            pcie_bandwidth_gbs: pcie_bw,
        }
    }).collect()
}

fn bench_ensemble(n: usize, rng: &mut ChaCha8Rng, std_sort_us: f64, hybrid_sort_us: f64) -> Vec<EnsembleEntry> {
    let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 1_000_000.0).collect();
    let mut ground_truth = data.clone();
    ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut entries = Vec::new();

    for k in [2, 3, 5, 11, 31, 101, 301, 1000] {
        let iterations = if k <= 31 { 20 } else if k <= 101 { 5 } else { 2 };
        let start = Instant::now();
        let mut confirmed_count = 0usize;
        let mut result = vec![0.0f64; 0];
        for _ in 0..iterations {
            let (r, c) = black_box(ensemble_voting_sort(&data, k));
            result = r;
            confirmed_count = c;
        }
        let sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;
        let (accuracy, _) = compute_accuracy(&ground_truth, &result);

        entries.push(EnsembleEntry {
            k,
            n,
            confirmed_pct: confirmed_count as f64 / n as f64 * 100.0,
            unconfirmed_count: n - confirmed_count,
            accuracy,
            sort_us,
            ns_per_elem: sort_us * 1000.0 / n as f64,
            speedup_vs_std: std_sort_us / sort_us,
            speedup_vs_hybrid: hybrid_sort_us / sort_us,
        });
    }
    entries
}

fn bench_multi_slit(n: usize, rng: &mut ChaCha8Rng, std_sort_us: f64) -> Vec<SlitEntry> {
    let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 1_000_000.0).collect();
    let mut ground_truth = data.clone();
    ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let iterations = if n <= 10_000 { 100 } else { 10 };
    let mut entries = Vec::new();

    for k in [1, 2, 3, 5, 7, 9, 11, 15] {
        let start = Instant::now();
        let mut result = vec![0.0f64; 0];
        for _ in 0..iterations {
            result = black_box(multi_slit_sort(&data, k));
        }
        let sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;
        let (accuracy, _) = compute_accuracy(&ground_truth, &result);

        entries.push(SlitEntry {
            k,
            n,
            accuracy,
            sort_us,
            ns_per_elem: sort_us * 1000.0 / n as f64,
            speedup_vs_std: std_sort_us / sort_us,
        });
    }
    entries
}

/// Count total inversions via bottom-up merge sort. O(n log n) time, O(n) space.
fn count_inversions(data: &[f64]) -> u64 {
    let n = data.len();
    let mut src: Vec<f64> = data.to_vec();
    let mut dst: Vec<f64> = vec![0.0; n];
    let mut inversions = 0u64;

    let mut width = 1usize;
    while width < n {
        let mut start = 0;
        while start < n {
            let mid = (start + width).min(n);
            let end = (start + 2 * width).min(n);

            let mut i = start;
            let mut j = mid;
            let mut k = start;

            while i < mid && j < end {
                if src[i] <= src[j] {
                    dst[k] = src[i];
                    i += 1;
                } else {
                    dst[k] = src[j];
                    inversions += (mid - i) as u64;
                    j += 1;
                }
                k += 1;
            }
            while i < mid { dst[k] = src[i]; i += 1; k += 1; }
            while j < end { dst[k] = src[j]; j += 1; k += 1; }

            start += 2 * width;
        }
        std::mem::swap(&mut src, &mut dst);
        width *= 2;
    }
    inversions
}

fn compute_large_vis(n: usize, rng: &mut ChaCha8Rng) -> LargeVisData {
    eprintln!("Generating large visualization data (n={})...", n);

    let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 1_000_000.0).collect();

    // Build table and scatter
    let table = ResonanceTable::build(&data, n);
    let scatter_result = table.sort_pure(&data);
    drop(table);

    let mut ground_truth = data.clone();
    ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    drop(data);

    // Exact match
    let mut exact = 0usize;
    for i in 0..n {
        if (ground_truth[i] - scatter_result[i]).abs() < 1e-9 { exact += 1; }
    }
    let exact_match_pct = exact as f64 / n as f64 * 100.0;

    // Adjacent inversions (O(n))
    let mut adj_inv = 0usize;
    for i in 0..n - 1 {
        if scatter_result[i] > scatter_result[i + 1] { adj_inv += 1; }
    }
    let adj_total = n - 1;
    let adj_order_pct = (1.0 - adj_inv as f64 / adj_total as f64) * 100.0;

    // Total inversions via merge sort (O(n log n)) — correct Kendall tau
    eprintln!("  Counting total inversions (merge sort)...");
    let total_inversions = count_inversions(&scatter_result);
    let total_pairs = (n as u64) * (n as u64 - 1) / 2;
    let kendall_tau_pct = (1.0 - 2.0 * total_inversions as f64 / total_pairs as f64) * 100.0;
    eprintln!("  inversions={} / {} pairs  kendall_tau={:.4}%", total_inversions, total_pairs, kendall_tau_pct);

    // Displacement stats via argsort (O(n log n))
    eprintln!("  Computing displacement stats (argsort)...");
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| scatter_result[a].partial_cmp(&scatter_result[b]).unwrap());
    let mut ranks = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank;
    }
    drop(indices);

    let mut total_disp: u64 = 0;
    let mut max_disp: usize = 0;
    let mut within2: usize = 0;
    for i in 0..n {
        let d = if ranks[i] >= i { ranks[i] - i } else { i - ranks[i] };
        total_disp += d as u64;
        if d > max_disp { max_disp = d; }
        if d <= 2 { within2 += 1; }
    }
    drop(ranks);

    let mean_displacement = total_disp as f64 / n as f64;
    let within_2_pct = within2 as f64 / n as f64 * 100.0;

    // Downsampled arrays (200 points)
    let n_down = 200;
    let step = n / n_down;
    let gt_downsampled: Vec<f64> = (0..n_down).map(|i| ground_truth[i * step]).collect();
    let scatter_downsampled: Vec<f64> = (0..n_down).map(|i| scatter_result[i * step]).collect();

    // Zoom region (50 points from middle)
    let zoom_start = n / 2;
    let zoom_n = 50;
    let zoom_gt: Vec<f64> = ground_truth[zoom_start..zoom_start + zoom_n].to_vec();
    let zoom_scatter: Vec<f64> = scatter_result[zoom_start..zoom_start + zoom_n].to_vec();

    eprintln!("  exact_match={:.2}%  kendall_tau={:.4}%  adj_order={:.2}%  mean_disp={:.2}  within2={:.1}%",
        exact_match_pct, kendall_tau_pct, adj_order_pct, mean_displacement, within_2_pct);

    LargeVisData {
        n,
        exact_match_pct,
        total_inversions,
        total_pairs,
        kendall_tau_pct,
        adj_inversions: adj_inv,
        adj_total,
        adj_order_pct,
        mean_displacement,
        max_displacement: max_disp,
        within_2_pct,
        gt_downsampled,
        scatter_downsampled,
        zoom_gt,
        zoom_scatter,
        zoom_start_index: zoom_start,
    }
}

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let sizes: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];

    let mut benchmarks = Vec::new();
    for &n in sizes {
        eprintln!("Benchmarking n={}...", n);
        let entry = bench_one(n, &mut rng);
        eprintln!(
            "  std={:.1}us  pure={:.1}us({:.1}%)  hybrid={:.1}us({:.1}%)  radix={:.1}us  [scatter={:.1}us bucket={:.1}us]",
            entry.std_sort_us,
            entry.pure_sort_us, entry.pure_accuracy * 100.0,
            entry.hybrid_sort_us, entry.hybrid_accuracy * 100.0,
            entry.radix_sort_us,
            entry.scatter_phase_us, entry.bucket_sort_phase_us,
        );
        benchmarks.push(entry);
    }

    // GPU+CPU heterogeneous estimate
    eprintln!("\n=== GPU+CPU Heterogeneous Estimate (GPU=500GB/s, CPU=50GB/s, PCIe=25GB/s) ===");
    let heterogeneous = bench_heterogeneous(&benchmarks);
    for h in &heterogeneous {
        eprintln!(
            "  n={:<9}  GPU+CPU={:>9.1}us  CPU hybrid={:>9.1}us  GPU radix={:>9.1}us  | vs std={:.1}x  vs cpu_hybrid={:.1}x  vs gpu_radix={:.1}x",
            h.n,
            h.hetero_total_us,
            h.cpu_hybrid_us,
            h.gpu_radix_us,
            h.hetero_vs_std,
            h.hetero_vs_cpu_hybrid,
            h.hetero_vs_gpu_radix,
        );
    }

    // === 3-way comparison: Hybrid vs Radix vs Ensemble k=1000 (all CPU) ===
    eprintln!("\n=== 3-Way CPU Comparison: Hybrid vs Radix vs Ensemble(k=1000) ===");
    let mut ensemble_sweep = Vec::new();
    let k_val = 1000;
    for &n in &[100usize, 1_000, 10_000, 100_000] {
        eprintln!("  Ensemble k={} n={}...", k_val, n);
        let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 1_000_000.0).collect();
        let mut ground_truth = data.clone();
        ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let iterations = if n <= 1_000 { 5 } else if n <= 10_000 { 2 } else { 1 };
        let start = Instant::now();
        let mut confirmed_count = 0usize;
        let mut result = vec![0.0f64; 0];
        for _ in 0..iterations {
            let (r, c) = black_box(ensemble_voting_sort(&data, k_val));
            result = r;
            confirmed_count = c;
        }
        let sort_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;
        let (accuracy, _) = compute_accuracy(&ground_truth, &result);

        let b = benchmarks.iter().find(|b| b.n == n).unwrap();
        let entry = EnsembleEntry {
            k: k_val,
            n,
            confirmed_pct: confirmed_count as f64 / n as f64 * 100.0,
            unconfirmed_count: n - confirmed_count,
            accuracy,
            sort_us,
            ns_per_elem: sort_us * 1000.0 / n as f64,
            speedup_vs_std: b.std_sort_us / sort_us,
            speedup_vs_hybrid: b.hybrid_sort_us / sort_us,
        };
        eprintln!(
            "    n={:<7}  Hybrid={:>10.1}us  Radix={:>10.1}us  Ensemble={:>10.1}us  | acc={:.4}  confirmed={:.1}%",
            n, b.hybrid_sort_us, b.radix_sort_us, sort_us, accuracy, entry.confirmed_pct,
        );
        ensemble_sweep.push(entry);
    }

    // Multi-slit sweep at n=10000
    let b10k = benchmarks.iter().find(|b| b.n == 10_000).unwrap();
    eprintln!("\nMulti-slit sweep (n=10000)...");
    let std_us_10k = b10k.std_sort_us;
    let multi_slit_sweep = bench_multi_slit(10_000, &mut rng, std_us_10k);
    for e in &multi_slit_sweep {
        eprintln!("  k={:>2}  acc={:.4}  sort={:.1}us  ns/elem={:.1}  speedup={:.2}x",
            e.k, e.accuracy, e.sort_us, e.ns_per_elem, e.speedup_vs_std);
    }

    // Visualization data
    eprintln!("\nGenerating visualization data (n=10000)...");
    let vis_n: usize = 10_000;
    let vis_data: Vec<f64> = (0..vis_n).map(|_| rng.gen::<f64>() * 1_000_000.0).collect();
    let table = ResonanceTable::build(&vis_data, vis_n);
    let pure_result = table.sort_pure(&vis_data);
    let hybrid_result = table.sort_hybrid(&vis_data);
    let mut ground_truth = vis_data.clone();
    ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let (_, pure_error_indices) = compute_accuracy(&ground_truth, &pure_result);
    let (_, hybrid_error_indices) = compute_accuracy(&ground_truth, &hybrid_result);

    let visualization = VisData {
        n: vis_n,
        ground_truth,
        pure_result,
        pure_error_indices,
        hybrid_result,
        hybrid_error_indices,
    };

    // Large-n visualization data (pre-computed stats + downsampled)
    let large_vis = Some(compute_large_vis(100_000_000, &mut rng));

    let result = BenchResult { benchmarks, heterogeneous, ensemble_sweep, multi_slit_sweep, visualization, large_vis };
    let json = serde_json::to_string_pretty(&result).unwrap();
    println!("{}", json);
    eprintln!("Done.");
}
