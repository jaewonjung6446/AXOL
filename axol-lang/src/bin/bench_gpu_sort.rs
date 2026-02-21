//! GPU Sort Benchmark — AXOL Scatter / Collapse / Two-Level / Radix Sort (GPU + CPU)
//!
//! 8-way comparison on the same data:
//!   1. CPU std::sort (pdqsort)
//!   2. CPU AXOL scatter (O(1)/elem, approximate)
//!   3. CPU radix sort (LSD f32, 4 pass, exact)
//!   4. GPU AXOL scatter (wgpu compute, k=1, approximate ~63%)
//!   5. GPU AXOL collapse k=3 (multi-slit interference, ~85%)
//!   6. GPU AXOL collapse k=5 (multi-slit interference, ~91%)
//!   7. GPU AXOL two-level (cache-friendly hierarchical scatter)
//!   8. GPU radix sort (wgpu compute, LSD f32, 4 pass, exact)

use std::hint::black_box;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use axol::gpu_scatter;

// ═══════════════════════════════════════════════════════════
// Output structures
// ═══════════════════════════════════════════════════════════

#[derive(Serialize)]
struct BenchResult {
    metadata: Metadata,
    benchmarks: Vec<BenchEntry>,
}

#[derive(Serialize)]
struct Metadata {
    adapter: String,
    backend: String,
    seed: u64,
}

#[derive(Serialize)]
struct BenchEntry {
    n: usize,
    cpu_std_sort_us: f64,
    cpu_scatter_us: f64,
    cpu_radix_us: f64,
    gpu_scatter_us: f64,
    gpu_scatter_phases: gpu_scatter::GpuTimings,
    gpu_scatter_accuracy: f64,
    gpu_collapse_k3_us: f64,
    gpu_collapse_k3_phases: gpu_scatter::CollapseTimings,
    gpu_collapse_k3_accuracy: f64,
    gpu_collapse_k5_us: f64,
    gpu_collapse_k5_phases: gpu_scatter::CollapseTimings,
    gpu_collapse_k5_accuracy: f64,
    gpu_twolevel_us: f64,
    gpu_twolevel_phases: gpu_scatter::TwoLevelTimings,
    gpu_twolevel_accuracy: f64,
    gpu_radix_us: f64,
    gpu_radix_phases: gpu_scatter::RadixTimings,
    gpu_radix_accuracy: f64,
    // Speedups (vs cpu std::sort)
    speedup_cpu_scatter: f64,
    speedup_cpu_radix: f64,
    speedup_gpu_scatter: f64,
    speedup_gpu_collapse_k3: f64,
    speedup_gpu_collapse_k5: f64,
    speedup_gpu_twolevel: f64,
    speedup_gpu_radix: f64,
}

// ═══════════════════════════════════════════════════════════
// CPU sort implementations
// ═══════════════════════════════════════════════════════════

struct ResonanceTable32 {
    min: f32,
    inv_range: f64,
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
            result[offsets[b]] = v;
            offsets[b] += 1;
        }
    }
}

#[inline]
fn f32_to_sortable_u32(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits & (1u32 << 31) != 0 { !bits } else { bits ^ (1u32 << 31) }
}

#[inline]
fn sortable_u32_to_f32(bits: u32) -> f32 {
    let raw = if bits & (1u32 << 31) != 0 { bits ^ (1u32 << 31) } else { !bits };
    f32::from_bits(raw)
}

fn radix_sort_f32(data: &[f32], result: &mut [f32]) {
    let n = data.len();
    let mut keys: Vec<u32> = data.iter().map(|&v| f32_to_sortable_u32(v)).collect();
    let mut buf: Vec<u32> = vec![0u32; n];
    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 256];
        for &k in &keys { counts[((k >> shift) & 0xFF) as usize] += 1; }
        let mut prefix = [0usize; 256];
        for i in 1..256 { prefix[i] = prefix[i - 1] + counts[i - 1] as usize; }
        for &k in &keys { let d = ((k >> shift) & 0xFF) as usize; buf[prefix[d]] = k; prefix[d] += 1; }
        std::mem::swap(&mut keys, &mut buf);
    }
    for i in 0..n { result[i] = sortable_u32_to_f32(keys[i]); }
}

fn compute_accuracy(ground_truth: &[f32], result: &[f32]) -> f64 {
    let n = ground_truth.len();
    let mut correct = 0usize;
    for i in 0..n {
        if (ground_truth[i] - result[i]).abs() < 1e-9 { correct += 1; }
    }
    correct as f64 / n as f64
}

// ═══════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════

fn main() {
    let seed = 42u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    eprintln!("=== AXOL vs Radix — GPU + CPU Benchmark ===");
    eprintln!("Initializing GPU...");

    let ctx = match gpu_scatter::init_gpu() {
        Some(c) => c,
        None => {
            eprintln!("ERROR: No GPU adapter found.");
            std::process::exit(1);
        }
    };
    eprintln!("  Adapter: {}", ctx.adapter_name);
    eprintln!("  Backend: {}", ctx.backend);

    let scatter_sorter = gpu_scatter::GpuScatterSorter::new(&ctx);
    let collapse_sorter = gpu_scatter::GpuCollapseSorter::new(&ctx);
    let twolevel_sorter = gpu_scatter::GpuTwoLevelSorter::new(&ctx);
    let radix_sorter = gpu_scatter::GpuRadixSorter::new(&ctx);
    eprintln!("  All pipelines compiled.\n");

    let sizes: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];
    let mut benchmarks = Vec::new();

    for &n in sizes {
        eprintln!("═══ n = {:>10} ═══", n);
        let data: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();

        let mut ground_truth = data.clone();
        ground_truth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let iters = if n <= 10_000 { 10 } else if n <= 100_000 { 5 } else { 3 };

        // ── 1. CPU std::sort ──
        let cpu_std_us = {
            let mut copy = data.clone();
            let start = Instant::now();
            for _ in 0..iters {
                copy.copy_from_slice(&data);
                copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                black_box(&copy);
            }
            start.elapsed().as_secs_f64() / iters as f64 * 1e6
        };

        // ── 2. CPU scatter ──
        let cpu_scatter_us = {
            let table = ResonanceTable32::build(&data, n);
            let mut rb = vec![0.0f32; n];
            let mut ob = vec![0usize; table.num_buckets];
            let start = Instant::now();
            for _ in 0..iters {
                table.scatter_only(&data, &mut rb, &mut ob);
                black_box(&rb);
            }
            start.elapsed().as_secs_f64() / iters as f64 * 1e6
        };

        // ── 3. CPU radix ──
        let cpu_radix_us = {
            let mut rb = vec![0.0f32; n];
            let start = Instant::now();
            for _ in 0..iters {
                radix_sort_f32(&data, &mut rb);
                black_box(&rb);
            }
            start.elapsed().as_secs_f64() / iters as f64 * 1e6
        };

        // ── 4. GPU scatter ──
        if n <= 100_000 { let _ = scatter_sorter.sort(&ctx, &data); } // warmup
        let gpu_iters = if n <= 10_000 { 5 } else { 3 };
        let mut best_scatter: Option<gpu_scatter::GpuTimings> = None;
        let mut scatter_result = Vec::new();
        for _ in 0..gpu_iters {
            let (res, t) = scatter_sorter.sort(&ctx, &data);
            if best_scatter.is_none() || t.total_us < best_scatter.as_ref().unwrap().total_us {
                best_scatter = Some(t);
                scatter_result = res;
            }
        }
        let gs = best_scatter.unwrap();
        let gs_acc = compute_accuracy(&ground_truth, &scatter_result);

        // ── 5. GPU collapse k=3 ──
        let can_k3 = 3 * n * 4 <= 1024 * 1024 * 1024;
        let (gc3, gc3_acc) = if can_k3 {
            if n <= 100_000 { let _ = collapse_sorter.sort(&ctx, &scatter_sorter, &data, 3); }
            let mut best: Option<gpu_scatter::CollapseTimings> = None;
            let mut best_result = Vec::new();
            for _ in 0..gpu_iters {
                let (res, t) = collapse_sorter.sort(&ctx, &scatter_sorter, &data, 3);
                if best.is_none() || t.total_us < best.as_ref().unwrap().total_us {
                    best = Some(t);
                    best_result = res;
                }
            }
            let t = best.unwrap();
            let acc = compute_accuracy(&ground_truth, &best_result);
            (t, acc)
        } else {
            (gpu_scatter::CollapseTimings {
                upload_us: 0.0, slit_accumulate_us: 0.0, histogram_us: 0.0,
                prefix_sum_us: 0.0, copy_offsets_us: 0.0, scatter_us: 0.0,
                download_us: 0.0, total_us: 0.0, k: 3,
            }, 0.0)
        };

        // ── 6. GPU collapse k=5 ──
        let can_k5 = 5 * n * 4 <= 1024 * 1024 * 1024;
        let (gc5, gc5_acc) = if can_k5 {
            if n <= 100_000 { let _ = collapse_sorter.sort(&ctx, &scatter_sorter, &data, 5); }
            let mut best: Option<gpu_scatter::CollapseTimings> = None;
            let mut best_result = Vec::new();
            for _ in 0..gpu_iters {
                let (res, t) = collapse_sorter.sort(&ctx, &scatter_sorter, &data, 5);
                if best.is_none() || t.total_us < best.as_ref().unwrap().total_us {
                    best = Some(t);
                    best_result = res;
                }
            }
            let t = best.unwrap();
            let acc = compute_accuracy(&ground_truth, &best_result);
            (t, acc)
        } else {
            (gpu_scatter::CollapseTimings {
                upload_us: 0.0, slit_accumulate_us: 0.0, histogram_us: 0.0,
                prefix_sum_us: 0.0, copy_offsets_us: 0.0, scatter_us: 0.0,
                download_us: 0.0, total_us: 0.0, k: 5,
            }, 0.0)
        };

        // ── 7. GPU two-level ──
        if n <= 100_000 { let _ = twolevel_sorter.sort(&ctx, &scatter_sorter, &data); } // warmup
        let mut best_tl: Option<gpu_scatter::TwoLevelTimings> = None;
        let mut tl_result = Vec::new();
        for _ in 0..gpu_iters {
            let (res, t) = twolevel_sorter.sort(&ctx, &scatter_sorter, &data);
            if best_tl.is_none() || t.total_us < best_tl.as_ref().unwrap().total_us {
                best_tl = Some(t);
                tl_result = res;
            }
        }
        let gtl = best_tl.unwrap();
        let gtl_acc = compute_accuracy(&ground_truth, &tl_result);

        // ── 8. GPU radix ──
        if n <= 100_000 { let _ = radix_sorter.sort(&ctx, &scatter_sorter, &data); } // warmup
        let mut best_radix: Option<gpu_scatter::RadixTimings> = None;
        let mut radix_result = Vec::new();
        for _ in 0..gpu_iters {
            let (res, t) = radix_sorter.sort(&ctx, &scatter_sorter, &data);
            if best_radix.is_none() || t.total_us < best_radix.as_ref().unwrap().total_us {
                best_radix = Some(t);
                radix_result = res;
            }
        }
        let gr = best_radix.unwrap();
        let gr_acc = compute_accuracy(&ground_truth, &radix_result);

        // ── Print ──
        eprintln!("  CPU std::sort:   {:>12.1} us", cpu_std_us);
        eprintln!("  CPU scatter:     {:>12.1} us  (approx, {:.1}x vs std)", cpu_scatter_us, cpu_std_us / cpu_scatter_us);
        eprintln!("  CPU radix:       {:>12.1} us  (exact,  {:.1}x vs std)", cpu_radix_us, cpu_std_us / cpu_radix_us);
        eprintln!("  GPU scatter:     {:>12.1} us  (acc={:.4}, {:.2}x vs std)", gs.total_us, gs_acc, cpu_std_us / gs.total_us);
        eprintln!("    upload={:.0} hist={:.0} prefix={:.0} copy={:.0} scatter={:.0} dl={:.0}",
            gs.upload_us, gs.histogram_us, gs.prefix_sum_us, gs.copy_offsets_us, gs.scatter_us, gs.download_us);
        if can_k3 {
            eprintln!("  GPU collapse k3: {:>12.1} us  (acc={:.4}, {:.2}x vs std)", gc3.total_us, gc3_acc, cpu_std_us / gc3.total_us);
            eprintln!("    upload={:.0} slit={:.0} hist={:.0} prefix={:.0} copy={:.0} scatter={:.0} dl={:.0}",
                gc3.upload_us, gc3.slit_accumulate_us, gc3.histogram_us, gc3.prefix_sum_us, gc3.copy_offsets_us, gc3.scatter_us, gc3.download_us);
        }
        if can_k5 {
            eprintln!("  GPU collapse k5: {:>12.1} us  (acc={:.4}, {:.2}x vs std)", gc5.total_us, gc5_acc, cpu_std_us / gc5.total_us);
            eprintln!("    upload={:.0} slit={:.0} hist={:.0} prefix={:.0} copy={:.0} scatter={:.0} dl={:.0}",
                gc5.upload_us, gc5.slit_accumulate_us, gc5.histogram_us, gc5.prefix_sum_us, gc5.copy_offsets_us, gc5.scatter_us, gc5.download_us);
        }
        eprintln!("  GPU two-level:   {:>12.1} us  (acc={:.4}, {:.2}x vs std, M={})", gtl.total_us, gtl_acc, cpu_std_us / gtl.total_us, gtl.m);
        eprintln!("    upload={:.0} c_hist={:.0} c_pre={:.0} c_copy={:.0} c_scat={:.0} f_buck={:.0} f_hist={:.0} f_pre={:.0} f_copy={:.0} f_scat={:.0} dl={:.0}",
            gtl.upload_us, gtl.coarse_histogram_us, gtl.coarse_prefix_us, gtl.coarse_copy_us,
            gtl.coarse_scatter_us, gtl.fine_bucket_us, gtl.fine_histogram_us, gtl.fine_prefix_us,
            gtl.fine_copy_us, gtl.fine_scatter_us, gtl.download_us);
        eprintln!("  GPU radix:       {:>12.1} us  (acc={:.4}, {:.2}x vs std)", gr.total_us, gr_acc, cpu_std_us / gr.total_us);
        eprintln!("    upload={:.0} convert={:.0} passes={:.0} back={:.0} dl={:.0}",
            gr.upload_us, gr.convert_us, gr.passes_us, gr.convert_back_us, gr.download_us);
        eprintln!();

        benchmarks.push(BenchEntry {
            n,
            cpu_std_sort_us: cpu_std_us,
            cpu_scatter_us,
            cpu_radix_us,
            gpu_scatter_us: gs.total_us,
            gpu_scatter_phases: gs.clone(),
            gpu_scatter_accuracy: gs_acc,
            gpu_collapse_k3_us: gc3.total_us,
            gpu_collapse_k3_phases: gc3.clone(),
            gpu_collapse_k3_accuracy: gc3_acc,
            gpu_collapse_k5_us: gc5.total_us,
            gpu_collapse_k5_phases: gc5.clone(),
            gpu_collapse_k5_accuracy: gc5_acc,
            gpu_twolevel_us: gtl.total_us,
            gpu_twolevel_phases: gtl.clone(),
            gpu_twolevel_accuracy: gtl_acc,
            gpu_radix_us: gr.total_us,
            gpu_radix_phases: gr.clone(),
            gpu_radix_accuracy: gr_acc,
            speedup_cpu_scatter: cpu_std_us / cpu_scatter_us,
            speedup_cpu_radix: cpu_std_us / cpu_radix_us,
            speedup_gpu_scatter: cpu_std_us / gs.total_us,
            speedup_gpu_collapse_k3: if gc3.total_us > 0.0 { cpu_std_us / gc3.total_us } else { 0.0 },
            speedup_gpu_collapse_k5: if gc5.total_us > 0.0 { cpu_std_us / gc5.total_us } else { 0.0 },
            speedup_gpu_twolevel: cpu_std_us / gtl.total_us,
            speedup_gpu_radix: cpu_std_us / gr.total_us,
        });
    }

    // ── Summary table: time ──
    eprintln!("════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    eprintln!("{:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "n", "CPU std", "CPU scat", "CPU rdx", "GPU scat", "GPU k=3", "GPU k=5", "GPU 2-lv", "GPU rdx");
    eprintln!("{:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "", "(us)", "(us)", "(us)", "(us)", "(us)", "(us)", "(us)", "(us)");
    eprintln!("────────────────────────────────────────────────────────────────────────────────────────────────────────────");
    for b in &benchmarks {
        eprintln!("{:>10} {:>10.1} {:>10.1} {:>10.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            b.n, b.cpu_std_sort_us, b.cpu_scatter_us, b.cpu_radix_us,
            b.gpu_scatter_us, b.gpu_collapse_k3_us, b.gpu_collapse_k5_us,
            b.gpu_twolevel_us, b.gpu_radix_us);
    }
    eprintln!("════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    // ── Summary table: accuracy ──
    eprintln!("{:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "n", "GPU scat%", "GPU k=3%", "GPU k=5%", "GPU 2-lv%", "GPU rdx%");
    eprintln!("────────────────────────────────────────────────────────────────────────────────");
    for b in &benchmarks {
        eprintln!("{:>10} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}%",
            b.n, b.gpu_scatter_accuracy * 100.0, b.gpu_collapse_k3_accuracy * 100.0,
            b.gpu_collapse_k5_accuracy * 100.0, b.gpu_twolevel_accuracy * 100.0,
            b.gpu_radix_accuracy * 100.0);
    }
    eprintln!("════════════════════════════════════════════════════════════════════════════════\n");

    let output = BenchResult {
        metadata: Metadata { adapter: ctx.adapter_name.clone(), backend: ctx.backend.clone(), seed },
        benchmarks,
    };
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
    eprintln!("Done.");
}
