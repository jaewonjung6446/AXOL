//! GPU Scatter Sort — wgpu compute shader implementation of AXOL Resonance Sort.
//!
//! Contains three GPU sorters:
//!   - `GpuScatterSorter`: Single-slit O(1) scatter (~63% accuracy)
//!   - `GpuCollapseSorter`: k-Slit Resonance Collapse Sort (AXOL multi-slit interference)
//!   - `GpuRadixSorter`: LSD radix sort (4 passes × 8-bit)
//!
//! The Collapse Sort uses AXOL's quantum-inspired framework:
//!   Wave (superposition) → k slit measurements → constructive interference
//!   → focus (partial collapse) → dephasing (collision resolution) → observe

use std::time::Instant;
use wgpu::util::DeviceExt;

// ═══════════════════════════════════════════════════════════
// Public types
// ═══════════════════════════════════════════════════════════

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
}

pub struct GpuScatterSorter {
    histogram_pipeline: wgpu::ComputePipeline,
    prefix_scan_pipeline: wgpu::ComputePipeline,
    prefix_add_pipeline: wgpu::ComputePipeline,
    copy_offsets_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    prefix_bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuTimings {
    pub upload_us: f64,
    pub histogram_us: f64,
    pub prefix_sum_us: f64,
    pub copy_offsets_us: f64,
    pub scatter_us: f64,
    pub download_us: f64,
    pub total_us: f64,
}

// ═══════════════════════════════════════════════════════════
// Params uniform (shared by histogram, scatter, copy_offsets)
// ═══════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
}

// Prefix sum params
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefixParams {
    n: u32,
    workgroups_x: u32,
    _pad1: u32,
    _pad2: u32,
}

// ═══════════════════════════════════════════════════════════
// WGSL Shaders
// ═══════════════════════════════════════════════════════════

const HISTOGRAM_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let val = data[idx];
    let bucket = min(u32((val - params.min_val) * params.inv_range), params.n - 1u);
    atomicAdd(&histogram[bucket], 1u);
}
"#;

const PREFIX_SCAN_WGSL: &str = r#"
struct PrefixParams {
    n: u32,
    workgroups_x: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PrefixParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let block_size = 512u;
    let linear_wid = wid.x + wid.y * params.workgroups_x;
    let base = linear_wid * block_size;
    let t = lid.x;

    // Load two elements per thread into shared memory
    let i0 = base + t;
    let i1 = base + t + 256u;
    temp[t] = select(0u, data[i0], i0 < params.n);
    temp[t + 256u] = select(0u, data[i1], i1 < params.n);

    // Up-sweep (reduce)
    var offset = 1u;
    var d = block_size >> 1u;
    loop {
        workgroupBarrier();
        if (t < d) {
            let ai = offset * (2u * t + 1u) - 1u;
            let bi = offset * (2u * t + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }
        offset = offset << 1u;
        d = d >> 1u;
        if (d == 0u) { break; }
    }

    // Store block total and clear last element
    workgroupBarrier();
    if (t == 0u) {
        block_sums[linear_wid] = temp[block_size - 1u];
        temp[block_size - 1u] = 0u;
    }

    // Down-sweep
    d = 1u;
    offset = block_size >> 1u;
    loop {
        workgroupBarrier();
        if (t < d) {
            let ai = offset * (2u * t + 1u) - 1u;
            let bi = offset * (2u * t + 2u) - 1u;
            let tmp = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + tmp;
        }
        d = d << 1u;
        offset = offset >> 1u;
        if (offset == 0u) { break; }
    }

    // Write back
    workgroupBarrier();
    if (i0 < params.n) { data[i0] = temp[t]; }
    if (i1 < params.n) { data[i1] = temp[t + 256u]; }
}
"#;

const PREFIX_ADD_WGSL: &str = r#"
struct PrefixParams {
    n: u32,
    workgroups_x: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PrefixParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read> block_sums: array<u32>;

@compute @workgroup_size(256)
fn add_block_sums(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let scan_block = idx / 512u;
    if (scan_block == 0u) { return; }
    data[idx] = data[idx] + block_sums[scan_block];
}
"#;

const COPY_OFFSETS_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    atomicStore(&offsets[idx], prefix_sum[idx]);
}
"#;

const SCATTER_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let val = data[idx];
    let bucket = min(u32((val - params.min_val) * params.inv_range), params.n - 1u);
    let pos = atomicAdd(&offsets[bucket], 1u);
    result[pos] = val;
}
"#;

// ═══════════════════════════════════════════════════════════
// GPU initialization
// ═══════════════════════════════════════════════════════════

pub fn init_gpu() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).ok()?;

    let adapter_name = adapter.get_info().name.clone();
    let backend = format!("{:?}", adapter.get_info().backend);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("axol_gpu_scatter"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_buffer_size: 2u64 * 1024 * 1024 * 1024, // 2GB
                max_storage_buffer_binding_size: 1024 * 1024 * 1024, // 1GB
                max_compute_workgroups_per_dimension: 65535,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        },
    )).ok()?;

    Some(GpuContext { device, queue, adapter_name, backend })
}

// ═══════════════════════════════════════════════════════════
// GpuScatterSorter implementation
// ═══════════════════════════════════════════════════════════

fn submit_and_wait(ctx: &GpuContext, encoder: wgpu::CommandEncoder) {
    let idx = ctx.queue.submit(std::iter::once(encoder.finish()));
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: None,
    });
}

impl GpuScatterSorter {
    pub fn new(ctx: &GpuContext) -> Self {
        // Bind group layout for histogram, copy_offsets (3 bindings: uniform, storage_ro, storage_rw)
        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Scatter needs 4 bindings (uniform, data_ro, offsets_rw, result_rw)
        let scatter_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_4_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Prefix sum bind group layout (uniform, data_rw, block_sums_rw)
        let prefix_bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Prefix add layout: binding 2 is read_only
        let prefix_add_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_add_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let make_pipeline = |label: &str, source: &str, entry: &str, layout: &wgpu::BindGroupLayout| -> wgpu::ComputePipeline {
            let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                immediate_size: 0,
            });
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let histogram_pipeline = make_pipeline("histogram", HISTOGRAM_WGSL, "main", &bind_group_layout);
        let prefix_scan_pipeline = make_pipeline("prefix_scan", PREFIX_SCAN_WGSL, "scan", &prefix_bind_group_layout);
        let prefix_add_pipeline = make_pipeline("prefix_add", PREFIX_ADD_WGSL, "add_block_sums", &prefix_add_bgl);
        let copy_offsets_pipeline = make_pipeline("copy_offsets", COPY_OFFSETS_WGSL, "main", &bind_group_layout);
        let scatter_pipeline = make_pipeline("scatter", SCATTER_WGSL, "main", &scatter_bgl);

        GpuScatterSorter {
            histogram_pipeline,
            prefix_scan_pipeline,
            prefix_add_pipeline,
            copy_offsets_pipeline,
            scatter_pipeline,
            bind_group_layout,
            prefix_bind_group_layout,
        }
    }

    pub fn sort(&self, ctx: &GpuContext, data: &[f32]) -> (Vec<f32>, GpuTimings) {
        let n = data.len();
        assert!(n > 0, "empty input");

        // CPU-side min/max
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in data {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let range = if (max_val - min_val).abs() < 1e-10 { 1.0f64 } else { (max_val - min_val) as f64 };
        let inv_range = (n as f64 - 1.0) / range;

        let wg = 256u32;
        let total_wg = (n as u32 + wg - 1) / wg;
        // 2D dispatch to stay within 65535 per dimension
        let dispatch_x = total_wg.min(65535);
        let dispatch_y = (total_wg + dispatch_x - 1) / dispatch_x;

        let params = Params {
            n: n as u32,
            min_val,
            inv_range: inv_range as f32,
            workgroups_x: dispatch_x,
        };

        // ── Upload ──
        let t_upload = Instant::now();

        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let data_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("data"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let zeros = vec![0u32; n];
        let histogram_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("histogram"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offsets"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let upload_us = t_upload.elapsed().as_secs_f64() * 1e6;

        // ── Dispatch 1: Histogram ──
        let t_hist = Instant::now();

        let hist_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: histogram_buf.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("histogram_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &hist_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        submit_and_wait(ctx, encoder);

        let histogram_us = t_hist.elapsed().as_secs_f64() * 1e6;

        // ── Dispatch 2: Prefix Sum (multi-level Blelloch) ──
        let t_prefix = Instant::now();
        self.gpu_prefix_sum(ctx, &histogram_buf, n);
        let prefix_sum_us = t_prefix.elapsed().as_secs_f64() * 1e6;

        // ── Dispatch 3: Copy offsets ──
        let t_copy = Instant::now();

        let copy_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: histogram_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: offsets_buf.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("copy_offsets"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_offsets_pipeline);
            pass.set_bind_group(0, &copy_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        submit_and_wait(ctx, encoder);

        let copy_offsets_us = t_copy.elapsed().as_secs_f64() * 1e6;

        // ── Dispatch 4: Scatter ──
        let t_scatter = Instant::now();

        // Scatter needs its own layout with 4 bindings
        let scatter_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_4_bgl_rt"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let scatter_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_bg"),
            layout: &scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: result_buf.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &scatter_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        submit_and_wait(ctx, encoder);

        let scatter_us = t_scatter.elapsed().as_secs_f64() * 1e6;

        // ── Download ──
        let t_download = Instant::now();

        let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_enc"),
        });
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging_buf, 0, (n * 4) as u64);
        let sub_idx = ctx.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(sub_idx),
            timeout: None,
        });
        receiver.recv().unwrap().unwrap();

        let mapped = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging_buf.unmap();

        let download_us = t_download.elapsed().as_secs_f64() * 1e6;

        let total_us = upload_us + histogram_us + prefix_sum_us + copy_offsets_us + scatter_us + download_us;

        let timings = GpuTimings {
            upload_us,
            histogram_us,
            prefix_sum_us,
            copy_offsets_us,
            scatter_us,
            download_us,
            total_us,
        };

        (result, timings)
    }

    /// Multi-level Blelloch exclusive prefix sum on a GPU buffer.
    pub fn gpu_prefix_sum(&self, ctx: &GpuContext, buf: &wgpu::Buffer, n: usize) {
        let block_size = 512u32; // 2 elements per thread, 256 threads
        let num_blocks = ((n as u32) + block_size - 1) / block_size;

        if num_blocks <= 1 {
            // Single block — just scan directly
            let block_sums_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("block_sums_L0"),
                size: 4, // 1 u32
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let pp = PrefixParams { n: n as u32, workgroups_x: 1, _pad1: 0, _pad2: 0 };
            let pp_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prefix_params"),
                contents: bytemuck::bytes_of(&pp),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scan_bg"),
                layout: &self.prefix_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: block_sums_buf.as_entire_binding() },
                ],
            });

            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.prefix_scan_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            submit_and_wait(ctx, encoder);
            return;
        }

        // Multi-level: scan blocks, recursively scan block sums, then add back
        let block_sums_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_sums"),
            size: (num_blocks as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clear block sums
        let zeros = vec![0u32; num_blocks as usize];
        ctx.queue.write_buffer(&block_sums_buf, 0, bytemuck::cast_slice(&zeros));

        // 2D dispatch for scan blocks
        let scan_dx = num_blocks.min(65535);
        let scan_dy = (num_blocks + scan_dx - 1) / scan_dx;

        let pp = PrefixParams { n: n as u32, workgroups_x: scan_dx, _pad1: 0, _pad2: 0 };
        let pp_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefix_params"),
            contents: bytemuck::bytes_of(&pp),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Step 1: Local scan of each block
        let scan_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_bg"),
            layout: &self.prefix_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: block_sums_buf.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.prefix_scan_pipeline);
            pass.set_bind_group(0, &scan_bg, &[]);
            pass.dispatch_workgroups(scan_dx, scan_dy, 1);
        }
        submit_and_wait(ctx, encoder);

        // Step 2: Recursively scan the block sums
        self.gpu_prefix_sum(ctx, &block_sums_buf, num_blocks as usize);

        // Step 3: Add scanned block sums back to each block
        // Need prefix_add layout (binding 2 is read_only)
        let add_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("add_bgl_rt"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Dispatch one thread per element (256 threads per workgroup), 2D
        let add_total = (n as u32 + 255) / 256;
        let add_dx = add_total.min(65535);
        let add_dy = (add_total + add_dx - 1) / add_dx;

        // Add shader needs its own params with correct workgroups_x
        let add_pp = PrefixParams { n: n as u32, workgroups_x: add_dx, _pad1: 0, _pad2: 0 };
        let add_pp_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("add_prefix_params"),
            contents: bytemuck::bytes_of(&add_pp),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let add_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_bg"),
            layout: &add_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: add_pp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: block_sums_buf.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.prefix_add_pipeline);
            pass.set_bind_group(0, &add_bg, &[]);
            pass.dispatch_workgroups(add_dx, add_dy, 1);
        }
        submit_and_wait(ctx, encoder);
    }
}

// ═══════════════════════════════════════════════════════════
// GPU Collapse Sort — k-Slit Resonance Collapse (AXOL)
// ═══════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CollapseParams {
    n: u32,
    min_val: f32,
    inv_range: f32,       // n / range (OffsetTable style)
    workgroups_x: u32,
    k: u32,               // number of slits
    kn: u32,              // k * n (histogram size)
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CollapseTimings {
    pub upload_us: f64,
    pub slit_accumulate_us: f64,
    pub histogram_us: f64,
    pub prefix_sum_us: f64,
    pub copy_offsets_us: f64,
    pub scatter_us: f64,
    pub download_us: f64,
    pub total_us: f64,
    pub k: u32,
}

// ── WGSL: Slit Accumulate ──
// Each thread computes k bucket assignments (one per slit offset)
// and writes the sum to pos_sum[idx]. No atomic needed (unique idx per thread).
const SLIT_ACCUMULATE_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
    k: u32,
    kn: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_sum: array<u32>;
@group(0) @binding(3) var<storage, read> slit_offsets: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let val = data[idx];
    var sum = 0u;
    for (var s = 0u; s < params.k; s = s + 1u) {
        let raw = (val - params.min_val) * params.inv_range + slit_offsets[s];
        let bucket = min(u32(raw), params.n - 1u);
        sum = sum + bucket;
    }
    pos_sum[idx] = sum;
}
"#;

// ── WGSL: Collapse Histogram ──
// Builds histogram from pos_sum values into k*n buckets.
const COLLAPSE_HISTOGRAM_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
    k: u32,
    kn: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pos_sum: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let bucket = min(pos_sum[idx], params.kn - 1u);
    atomicAdd(&histogram[bucket], 1u);
}
"#;

// ── WGSL: Collapse Copy Offsets ──
// Copies prefix-summed histogram into atomic offsets buffer (for k*n elements).
const COLLAPSE_COPY_OFFSETS_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
    k: u32,
    kn: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.kn) { return; }
    atomicStore(&offsets[idx], prefix_sum[idx]);
}
"#;

// ── WGSL: Collapse Scatter ──
// Scatters elements to positions determined by their collapsed pos_sum.
const COLLAPSE_SCATTER_WGSL: &str = r#"
struct Params {
    n: u32,
    min_val: f32,
    inv_range: f32,
    workgroups_x: u32,
    k: u32,
    kn: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> pos_sum: array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let bucket = min(pos_sum[idx], params.kn - 1u);
    let pos = atomicAdd(&offsets[bucket], 1u);
    result[pos] = data[idx];
}
"#;

pub struct GpuCollapseSorter {
    slit_accumulate_pipeline: wgpu::ComputePipeline,
    collapse_histogram_pipeline: wgpu::ComputePipeline,
    collapse_copy_offsets_pipeline: wgpu::ComputePipeline,
    collapse_scatter_pipeline: wgpu::ComputePipeline,
    slit_bgl: wgpu::BindGroupLayout,         // uniform + data_ro + pos_sum_rw + slit_offsets_ro
    histogram_bgl: wgpu::BindGroupLayout,     // uniform + pos_sum_ro + histogram_rw
    copy_bgl: wgpu::BindGroupLayout,          // uniform + prefix_sum_ro + offsets_rw
    scatter_bgl: wgpu::BindGroupLayout,       // uniform + data_ro + pos_sum_ro + offsets_rw + result_rw
}

impl GpuCollapseSorter {
    pub fn new(ctx: &GpuContext) -> Self {
        // Helper to create a BGL entry
        let bgl_entry = |binding: u32, ty: wgpu::BufferBindingType| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }
        };
        let uniform = wgpu::BufferBindingType::Uniform;
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };

        // Slit accumulate: uniform + data(ro) + pos_sum(rw) + slit_offsets(ro)
        let slit_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slit_bgl"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, rw), bgl_entry(3, ro)],
        });

        // Collapse histogram: uniform + pos_sum(ro) + histogram(rw)
        let histogram_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collapse_hist_bgl"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, rw)],
        });

        // Copy offsets: uniform + prefix_sum(ro) + offsets(rw)
        let copy_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collapse_copy_bgl"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, rw)],
        });

        // Scatter: uniform + data(ro) + pos_sum(ro) + offsets(rw) + result(rw)
        let scatter_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collapse_scatter_bgl"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, ro), bgl_entry(3, rw), bgl_entry(4, rw)],
        });

        let make_pipeline = |label: &str, source: &str, entry: &str, layout: &wgpu::BindGroupLayout| -> wgpu::ComputePipeline {
            let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                immediate_size: 0,
            });
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pl), module: &shader,
                entry_point: Some(entry), compilation_options: Default::default(), cache: None,
            })
        };

        GpuCollapseSorter {
            slit_accumulate_pipeline: make_pipeline("slit_accum", SLIT_ACCUMULATE_WGSL, "main", &slit_bgl),
            collapse_histogram_pipeline: make_pipeline("collapse_hist", COLLAPSE_HISTOGRAM_WGSL, "main", &histogram_bgl),
            collapse_copy_offsets_pipeline: make_pipeline("collapse_copy", COLLAPSE_COPY_OFFSETS_WGSL, "main", &copy_bgl),
            collapse_scatter_pipeline: make_pipeline("collapse_scatter", COLLAPSE_SCATTER_WGSL, "main", &scatter_bgl),
            slit_bgl,
            histogram_bgl,
            copy_bgl,
            scatter_bgl,
        }
    }

    pub fn sort(
        &self,
        ctx: &GpuContext,
        scatter_sorter: &GpuScatterSorter,
        data: &[f32],
        k: u32,
    ) -> (Vec<f32>, CollapseTimings) {
        let n = data.len();
        assert!(n > 0, "empty input");
        assert!(k >= 1, "k must be >= 1");
        let kn = (k as usize) * n;
        assert!(kn * 4 <= 1024 * 1024 * 1024, "k*n exceeds 1GB storage buffer limit");

        // CPU-side min/max
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in data {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let range = if (max_val - min_val).abs() < 1e-10 { 1.0f64 } else { (max_val - min_val) as f64 };
        // OffsetTable style: n/range (not (n-1)/range)
        let inv_range = (n as f64) / range;

        let wg = 256u32;
        let total_wg = (n as u32 + wg - 1) / wg;
        let dispatch_x = total_wg.min(65535);
        let dispatch_y = (total_wg + dispatch_x - 1) / dispatch_x;

        // For k*n-sized buffer dispatches
        let kn_wg = (kn as u32 + wg - 1) / wg;
        let kn_dispatch_x = kn_wg.min(65535);
        let kn_dispatch_y = (kn_wg + kn_dispatch_x - 1) / kn_dispatch_x;

        let params = CollapseParams {
            n: n as u32,
            min_val,
            inv_range: inv_range as f32,
            workgroups_x: dispatch_x,
            k,
            kn: kn as u32,
            _pad1: 0,
            _pad2: 0,
        };

        // Params variant for k*n dispatches (copy_offsets uses kn-sized workgroups)
        let params_kn = CollapseParams {
            workgroups_x: kn_dispatch_x,
            ..params
        };

        // Slit offsets: [0/k, 1/k, 2/k, ...]
        let slit_offsets: Vec<f32> = (0..k).map(|s| s as f32 / k as f32).collect();

        // ── Upload ──
        let t_upload = Instant::now();

        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("collapse_params"), contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_kn_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("collapse_params_kn"), contents: bytemuck::bytes_of(&params_kn),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let data_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("collapse_data"), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pos_sum_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos_sum"), contents: bytemuck::cast_slice(&vec![0u32; n]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let slit_offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slit_offsets"), contents: bytemuck::cast_slice(&slit_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let histogram_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("collapse_hist"), contents: bytemuck::cast_slice(&vec![0u32; kn]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("collapse_offsets"), contents: bytemuck::cast_slice(&vec![0u32; kn]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collapse_result"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let upload_us = t_upload.elapsed().as_secs_f64() * 1e6;

        // ── Step 1: Slit Accumulate (single dispatch, k-loop in shader) ──
        let t_slit = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("slit_bg"), layout: &self.slit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pos_sum_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: slit_offsets_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.slit_accumulate_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let slit_accumulate_us = t_slit.elapsed().as_secs_f64() * 1e6;

        // ── Step 2: Collapse Histogram ──
        let t_hist = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chist_bg"), layout: &self.histogram_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pos_sum_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: histogram_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.collapse_histogram_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let histogram_us = t_hist.elapsed().as_secs_f64() * 1e6;

        // ── Step 3: Prefix Sum on k*n histogram ──
        let t_prefix = Instant::now();
        scatter_sorter.gpu_prefix_sum(ctx, &histogram_buf, kn);
        let prefix_sum_us = t_prefix.elapsed().as_secs_f64() * 1e6;

        // ── Step 4: Copy Offsets (k*n elements) ──
        let t_copy = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ccopy_bg"), layout: &self.copy_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_kn_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: histogram_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: offsets_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.collapse_copy_offsets_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(kn_dispatch_x, kn_dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let copy_offsets_us = t_copy.elapsed().as_secs_f64() * 1e6;

        // ── Step 5: Collapse Scatter ──
        let t_scatter = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cscatter_bg"), layout: &self.scatter_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pos_sum_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: offsets_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: result_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.collapse_scatter_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let scatter_us = t_scatter.elapsed().as_secs_f64() * 1e6;

        // ── Download ──
        let t_download = Instant::now();
        let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collapse_staging"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging_buf, 0, (n * 4) as u64);
        let sub_idx = ctx.queue.submit(std::iter::once(encoder.finish()));
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| { sender.send(r).unwrap(); });
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(sub_idx), timeout: None });
        receiver.recv().unwrap().unwrap();
        let mapped = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging_buf.unmap();
        let download_us = t_download.elapsed().as_secs_f64() * 1e6;

        let total_us = upload_us + slit_accumulate_us + histogram_us + prefix_sum_us
            + copy_offsets_us + scatter_us + download_us;

        let timings = CollapseTimings {
            upload_us, slit_accumulate_us, histogram_us, prefix_sum_us,
            copy_offsets_us, scatter_us, download_us, total_us, k,
        };

        (result, timings)
    }
}

// ═══════════════════════════════════════════════════════════
// GPU Radix Sort (LSD, f32, 4 passes × 8-bit digit)
// ═══════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RadixParams {
    n: u32,
    shift: u32,
    workgroups_x: u32,
    _pad: u32,
}

const RADIX_HISTOGRAM_WGSL: &str = r#"
struct RadixParams {
    n: u32,
    shift: u32,
    workgroups_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RadixParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let digit = (keys[idx] >> params.shift) & 0xFFu;
    atomicAdd(&histogram[digit], 1u);
}
"#;

const RADIX_SCATTER_WGSL: &str = r#"
struct RadixParams {
    n: u32,
    shift: u32,
    workgroups_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RadixParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let key = keys_in[idx];
    let digit = (key >> params.shift) & 0xFFu;
    let pos = atomicAdd(&offsets[digit], 1u);
    keys_out[pos] = key;
}
"#;

const RADIX_CONVERT_WGSL: &str = r#"
struct RadixParams {
    n: u32,
    shift: u32,
    workgroups_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RadixParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let bits = bitcast<u32>(data[idx]);
    // f32 total order: negative flip all, positive flip sign bit
    if ((bits & 0x80000000u) != 0u) {
        keys[idx] = ~bits;
    } else {
        keys[idx] = bits ^ 0x80000000u;
    }
}
"#;

const RADIX_CONVERT_BACK_WGSL: &str = r#"
struct RadixParams {
    n: u32,
    shift: u32,
    workgroups_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RadixParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let bits = keys[idx];
    // Reverse the total order encoding
    if ((bits & 0x80000000u) != 0u) {
        result[idx] = bitcast<f32>(bits ^ 0x80000000u);
    } else {
        result[idx] = bitcast<f32>(~bits);
    }
}
"#;

pub struct GpuRadixSorter {
    convert_pipeline: wgpu::ComputePipeline,
    histogram_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    convert_back_pipeline: wgpu::ComputePipeline,
    bgl_3: wgpu::BindGroupLayout,  // uniform + storage_ro + storage_rw
    bgl_4: wgpu::BindGroupLayout,  // uniform + storage_ro + storage_rw + storage_rw
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RadixTimings {
    pub upload_us: f64,
    pub convert_us: f64,
    pub passes_us: f64,
    pub convert_back_us: f64,
    pub download_us: f64,
    pub total_us: f64,
}

impl GpuRadixSorter {
    pub fn new(ctx: &GpuContext) -> Self {
        let bgl_3 = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_bgl_3"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let bgl_4 = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_bgl_4"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let make_pipeline = |label: &str, source: &str, entry: &str, layout: &wgpu::BindGroupLayout| -> wgpu::ComputePipeline {
            let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                immediate_size: 0,
            });
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pl), module: &shader,
                entry_point: Some(entry), compilation_options: Default::default(), cache: None,
            })
        };

        let convert_pipeline = make_pipeline("radix_convert", RADIX_CONVERT_WGSL, "main", &bgl_3);
        let histogram_pipeline = make_pipeline("radix_histogram", RADIX_HISTOGRAM_WGSL, "main", &bgl_3);
        let scatter_pipeline = make_pipeline("radix_scatter", RADIX_SCATTER_WGSL, "main", &bgl_4);
        let convert_back_pipeline = make_pipeline("radix_convert_back", RADIX_CONVERT_BACK_WGSL, "main", &bgl_3);

        GpuRadixSorter { convert_pipeline, histogram_pipeline, scatter_pipeline, convert_back_pipeline, bgl_3, bgl_4 }
    }

    pub fn sort(&self, ctx: &GpuContext, scatter_sorter: &GpuScatterSorter, data: &[f32]) -> (Vec<f32>, RadixTimings) {
        let n = data.len();
        let wg = 256u32;
        let total_wg = (n as u32 + wg - 1) / wg;
        let dx = total_wg.min(65535);
        let dy = (total_wg + dx - 1) / dx;

        // ── Upload ──
        let t_upload = Instant::now();

        let data_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix_data"), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let keys_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keys_a"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let keys_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keys_b"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let histogram_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_hist"), size: 256 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let offsets_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_offsets"), size: 256 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_result"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let upload_us = t_upload.elapsed().as_secs_f64() * 1e6;

        // ── Convert f32 → sortable u32 ──
        let t_convert = Instant::now();
        {
            let params = RadixParams { n: n as u32, shift: 0, workgroups_x: dx, _pad: 0 };
            let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rp"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("convert_bg"), layout: &self.bgl_3,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: keys_a.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.convert_pipeline); pass.set_bind_group(0, &bg, &[]);
              pass.dispatch_workgroups(dx, dy, 1); }
            submit_and_wait(ctx, encoder);
        }
        let convert_us = t_convert.elapsed().as_secs_f64() * 1e6;

        // ── 4 LSD passes ──
        let t_passes = Instant::now();
        let key_bufs = [&keys_a, &keys_b];
        for pass_idx in 0u32..4 {
            let shift = pass_idx * 8;
            let src = key_bufs[(pass_idx as usize) % 2];
            let dst = key_bufs[(pass_idx as usize + 1) % 2];

            // Clear histogram
            let zeros = [0u32; 256];
            ctx.queue.write_buffer(&histogram_buf, 0, bytemuck::cast_slice(&zeros));

            let params = RadixParams { n: n as u32, shift, workgroups_x: dx, _pad: 0 };
            let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rp"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
            });

            // Histogram
            let hist_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rh_bg"), layout: &self.bgl_3,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: histogram_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.histogram_pipeline); pass.set_bind_group(0, &hist_bg, &[]);
              pass.dispatch_workgroups(dx, dy, 1); }
            submit_and_wait(ctx, encoder);

            // Prefix sum on 256-element histogram (reuse scatter_sorter's infrastructure)
            scatter_sorter.gpu_prefix_sum(ctx, &histogram_buf, 256);

            // Copy prefix sum → offsets
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&histogram_buf, 0, &offsets_buf, 0, 256 * 4);
            submit_and_wait(ctx, encoder);

            // Scatter
            let scatter_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rs_bg"), layout: &self.bgl_4,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: offsets_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dst.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.scatter_pipeline); pass.set_bind_group(0, &scatter_bg, &[]);
              pass.dispatch_workgroups(dx, dy, 1); }
            submit_and_wait(ctx, encoder);
        }
        let passes_us = t_passes.elapsed().as_secs_f64() * 1e6;

        // After 4 passes, sorted keys are in keys_a (even number of swaps)
        let sorted_keys = &keys_a;

        // ── Convert back u32 → f32 ──
        let t_back = Instant::now();
        {
            let params = RadixParams { n: n as u32, shift: 0, workgroups_x: dx, _pad: 0 };
            let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rp"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("back_bg"), layout: &self.bgl_3,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sorted_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: result_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.convert_back_pipeline); pass.set_bind_group(0, &bg, &[]);
              pass.dispatch_workgroups(dx, dy, 1); }
            submit_and_wait(ctx, encoder);
        }
        let convert_back_us = t_back.elapsed().as_secs_f64() * 1e6;

        // ── Download ──
        let t_dl = Instant::now();
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_staging"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, (n * 4) as u64);
        let sub_idx = ctx.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| { sender.send(r).unwrap(); });
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(sub_idx), timeout: None });
        receiver.recv().unwrap().unwrap();

        let mapped = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        let download_us = t_dl.elapsed().as_secs_f64() * 1e6;

        let total_us = upload_us + convert_us + passes_us + convert_back_us + download_us;
        let timings = RadixTimings { upload_us, convert_us, passes_us, convert_back_us, download_us, total_us };
        (result, timings)
    }
}

// ═══════════════════════════════════════════════════════════
// GPU Two-Level AXOL Sort — Cache-friendly hierarchical scatter
// ═══════════════════════════════════════════════════════════
//
// Solves the n >= 10M cache thrashing problem:
//   Phase 1: Coarse scatter into √n buckets (histogram fits in shared memory)
//   Phase 2: Fine scatter within each bucket (localized memory access)
//
// Both phases use the same AXOL scatter principle, but Phase 1's histogram
// is only √n bins (fits in L1), and Phase 2's n-bin histogram access is
// localized because the data is coarse-sorted.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TwoLevelParams {
    n: u32,
    min_val: f32,
    inv_range_coarse: f32,   // M / range
    workgroups_x: u32,
    m: u32,                  // coarse bucket count = ceil(sqrt(n))
    range: f32,              // max - min
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TwoLevelTimings {
    pub upload_us: f64,
    pub coarse_histogram_us: f64,
    pub coarse_prefix_us: f64,
    pub coarse_copy_us: f64,
    pub coarse_scatter_us: f64,
    pub fine_bucket_us: f64,
    pub fine_histogram_us: f64,
    pub fine_prefix_us: f64,
    pub fine_copy_us: f64,
    pub fine_scatter_us: f64,
    pub download_us: f64,
    pub total_us: f64,
    pub m: u32,
}

// ── WGSL: Coarse Histogram ──
const TWOLEVEL_COARSE_HIST_WGSL: &str = r#"
struct Params {
    n: u32, min_val: f32, inv_range_coarse: f32, workgroups_x: u32,
    m: u32, range: f32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let val = data[idx];
    let bucket = min(u32((val - params.min_val) * params.inv_range_coarse), params.m - 1u);
    atomicAdd(&histogram[bucket], 1u);
}
"#;

// ── WGSL: Coarse Scatter ──
const TWOLEVEL_COARSE_SCATTER_WGSL: &str = r#"
struct Params {
    n: u32, min_val: f32, inv_range_coarse: f32, workgroups_x: u32,
    m: u32, range: f32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> mid: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let val = data[idx];
    let bucket = min(u32((val - params.min_val) * params.inv_range_coarse), params.m - 1u);
    let pos = atomicAdd(&offsets[bucket], 1u);
    mid[pos] = val;
}
"#;

// ── WGSL: Fine Bucket Computation ──
// Computes fine position within each coarse bucket using saved coarse offsets.
const TWOLEVEL_FINE_BUCKET_WGSL: &str = r#"
struct Params {
    n: u32, min_val: f32, inv_range_coarse: f32, workgroups_x: u32,
    m: u32, range: f32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mid: array<f32>;
@group(0) @binding(2) var<storage, read> coarse_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> fine_bucket: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let val = mid[idx];

    // Recompute coarse bucket from value
    let raw = (val - params.min_val) * params.inv_range_coarse;
    let coarse_id = min(u32(raw), params.m - 1u);

    // Bucket boundaries from saved prefix sums (M+1 elements)
    let bucket_start = coarse_offsets[coarse_id];
    let bucket_end = coarse_offsets[coarse_id + 1u];
    let bucket_count = bucket_end - bucket_start;

    if (bucket_count <= 1u) {
        fine_bucket[idx] = bucket_start;
        return;
    }

    // Fine position: fraction within coarse bucket × bucket element count
    let local_frac = clamp(raw - f32(coarse_id), 0.0, 0.999999);
    let fine_pos = min(u32(local_frac * f32(bucket_count)), bucket_count - 1u);
    fine_bucket[idx] = bucket_start + fine_pos;
}
"#;

// ── WGSL: Fine Histogram ──
const TWOLEVEL_FINE_HIST_WGSL: &str = r#"
struct Params {
    n: u32, min_val: f32, inv_range_coarse: f32, workgroups_x: u32,
    m: u32, range: f32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> fine_bucket: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let bucket = fine_bucket[idx];
    atomicAdd(&histogram[bucket], 1u);
}
"#;

// ── WGSL: Fine Scatter ──
const TWOLEVEL_FINE_SCATTER_WGSL: &str = r#"
struct Params {
    n: u32, min_val: f32, inv_range_coarse: f32, workgroups_x: u32,
    m: u32, range: f32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mid: array<f32>;
@group(0) @binding(2) var<storage, read> fine_bucket: array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.workgroups_x * 256u;
    if (idx >= params.n) { return; }
    let bucket = fine_bucket[idx];
    let pos = atomicAdd(&offsets[bucket], 1u);
    result[pos] = mid[idx];
}
"#;

pub struct GpuTwoLevelSorter {
    coarse_hist_pipeline: wgpu::ComputePipeline,
    coarse_scatter_pipeline: wgpu::ComputePipeline,
    fine_bucket_pipeline: wgpu::ComputePipeline,
    fine_hist_pipeline: wgpu::ComputePipeline,
    fine_scatter_pipeline: wgpu::ComputePipeline,
    bgl_3: wgpu::BindGroupLayout,
    bgl_4_scatter: wgpu::BindGroupLayout,
    bgl_4_fine: wgpu::BindGroupLayout,
    bgl_5_scatter: wgpu::BindGroupLayout,
}

impl GpuTwoLevelSorter {
    pub fn new(ctx: &GpuContext) -> Self {
        let bgl_entry = |binding: u32, ty: wgpu::BufferBindingType| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }
        };
        let uniform = wgpu::BufferBindingType::Uniform;
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };

        let bgl_3 = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tl_bgl_3"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, rw)],
        });
        let bgl_4_scatter = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tl_bgl_4s"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, rw), bgl_entry(3, rw)],
        });
        let bgl_4_fine = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tl_bgl_4f"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, ro), bgl_entry(3, rw)],
        });
        let bgl_5_scatter = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tl_bgl_5s"),
            entries: &[bgl_entry(0, uniform), bgl_entry(1, ro), bgl_entry(2, ro), bgl_entry(3, rw), bgl_entry(4, rw)],
        });

        let make_pipeline = |label: &str, source: &str, entry: &str, layout: &wgpu::BindGroupLayout| -> wgpu::ComputePipeline {
            let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label), bind_group_layouts: &[layout], immediate_size: 0,
            });
            ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pl), module: &shader,
                entry_point: Some(entry), compilation_options: Default::default(), cache: None,
            })
        };

        GpuTwoLevelSorter {
            coarse_hist_pipeline: make_pipeline("tl_chist", TWOLEVEL_COARSE_HIST_WGSL, "main", &bgl_3),
            coarse_scatter_pipeline: make_pipeline("tl_cscat", TWOLEVEL_COARSE_SCATTER_WGSL, "main", &bgl_4_scatter),
            fine_bucket_pipeline: make_pipeline("tl_fbucket", TWOLEVEL_FINE_BUCKET_WGSL, "main", &bgl_4_fine),
            fine_hist_pipeline: make_pipeline("tl_fhist", TWOLEVEL_FINE_HIST_WGSL, "main", &bgl_3),
            fine_scatter_pipeline: make_pipeline("tl_fscat", TWOLEVEL_FINE_SCATTER_WGSL, "main", &bgl_5_scatter),
            bgl_3, bgl_4_scatter, bgl_4_fine, bgl_5_scatter,
        }
    }

    pub fn sort(
        &self,
        ctx: &GpuContext,
        scatter_sorter: &GpuScatterSorter,
        data: &[f32],
    ) -> (Vec<f32>, TwoLevelTimings) {
        let n = data.len();
        assert!(n > 0);
        let m = ((n as f64).sqrt().ceil() as u32).max(2);
        let m_alloc = (m + 1) as usize; // M+1: sentinel for bucket_end of last bucket

        // CPU min/max
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in data {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let range_f64 = if (max_val - min_val).abs() < 1e-10 { 1.0f64 } else { (max_val - min_val) as f64 };
        let inv_range_coarse = m as f64 / range_f64;

        let wg = 256u32;
        let total_wg = (n as u32 + wg - 1) / wg;
        let dispatch_x = total_wg.min(65535);
        let dispatch_y = (total_wg + dispatch_x - 1) / dispatch_x;

        let params = TwoLevelParams {
            n: n as u32,
            min_val,
            inv_range_coarse: inv_range_coarse as f32,
            workgroups_x: dispatch_x,
            m,
            range: (max_val - min_val).max(1e-10),
            _pad1: 0, _pad2: 0,
        };

        // ── Upload ──
        let t_upload = Instant::now();

        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tl_params"), contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let data_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tl_data"), contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Coarse histogram: M+1 elements (last stays 0, becomes n after prefix sum)
        let coarse_hist_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tl_chist"),
            contents: bytemuck::cast_slice(&vec![0u32; m_alloc]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let coarse_offsets_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tl_coff"), size: (m_alloc * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Saved coarse prefix sums for Phase 2 (M+1 elements, read-only)
        let coarse_saved_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tl_csaved"), size: (m_alloc * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mid_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tl_mid"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fine_bucket_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tl_fb"),
            contents: bytemuck::cast_slice(&vec![0u32; n]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fine_hist_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tl_fhist"),
            contents: bytemuck::cast_slice(&vec![0u32; n]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let fine_offsets_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tl_foff"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tl_result"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let upload_us = t_upload.elapsed().as_secs_f64() * 1e6;

        // ═══ PHASE 1: Coarse Sort (√n bins — fits in shared memory) ═══

        // Step 1: Coarse Histogram
        let t_chist = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tl_chist_bg"), layout: &self.bgl_3,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: coarse_hist_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.coarse_hist_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let coarse_histogram_us = t_chist.elapsed().as_secs_f64() * 1e6;

        // Step 2: Prefix Sum on M+1 elements (tiny — single workgroup)
        let t_cpre = Instant::now();
        scatter_sorter.gpu_prefix_sum(ctx, &coarse_hist_buf, m_alloc);
        let coarse_prefix_us = t_cpre.elapsed().as_secs_f64() * 1e6;

        // Step 3: Copy offsets (save prefix sums for Phase 2 + scatter offsets)
        let t_ccopy = Instant::now();
        {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&coarse_hist_buf, 0, &coarse_saved_buf, 0, (m_alloc * 4) as u64);
            encoder.copy_buffer_to_buffer(&coarse_hist_buf, 0, &coarse_offsets_buf, 0, (m_alloc * 4) as u64);
            submit_and_wait(ctx, encoder);
        }
        let coarse_copy_us = t_ccopy.elapsed().as_secs_f64() * 1e6;

        // Step 4: Coarse Scatter
        let t_cscat = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tl_cscat_bg"), layout: &self.bgl_4_scatter,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: coarse_offsets_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: mid_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.coarse_scatter_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let coarse_scatter_us = t_cscat.elapsed().as_secs_f64() * 1e6;

        // ═══ PHASE 2: Fine Sort (n bins — but localized access after coarse sort) ═══

        // Step 5: Fine Bucket Computation
        let t_fbucket = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tl_fb_bg"), layout: &self.bgl_4_fine,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: mid_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: coarse_saved_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: fine_bucket_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fine_bucket_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let fine_bucket_us = t_fbucket.elapsed().as_secs_f64() * 1e6;

        // Step 6: Fine Histogram (n bins, but localized access)
        let t_fhist = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tl_fhist_bg"), layout: &self.bgl_3,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fine_bucket_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: fine_hist_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fine_hist_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let fine_histogram_us = t_fhist.elapsed().as_secs_f64() * 1e6;

        // Step 7: Prefix Sum on n elements
        let t_fpre = Instant::now();
        scatter_sorter.gpu_prefix_sum(ctx, &fine_hist_buf, n);
        let fine_prefix_us = t_fpre.elapsed().as_secs_f64() * 1e6;

        // Step 8: Copy fine offsets
        let t_fcopy = Instant::now();
        {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&fine_hist_buf, 0, &fine_offsets_buf, 0, (n * 4) as u64);
            submit_and_wait(ctx, encoder);
        }
        let fine_copy_us = t_fcopy.elapsed().as_secs_f64() * 1e6;

        // Step 9: Fine Scatter
        let t_fscat = Instant::now();
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tl_fscat_bg"), layout: &self.bgl_5_scatter,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: mid_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: fine_bucket_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: fine_offsets_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: result_buf.as_entire_binding() },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fine_scatter_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            submit_and_wait(ctx, encoder);
        }
        let fine_scatter_us = t_fscat.elapsed().as_secs_f64() * 1e6;

        // ── Download ──
        let t_download = Instant::now();
        let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tl_staging"), size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging_buf, 0, (n * 4) as u64);
        let sub_idx = ctx.queue.submit(std::iter::once(encoder.finish()));
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| { sender.send(r).unwrap(); });
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(sub_idx), timeout: None });
        receiver.recv().unwrap().unwrap();
        let mapped = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging_buf.unmap();
        let download_us = t_download.elapsed().as_secs_f64() * 1e6;

        let total_us = upload_us + coarse_histogram_us + coarse_prefix_us + coarse_copy_us
            + coarse_scatter_us + fine_bucket_us + fine_histogram_us + fine_prefix_us
            + fine_copy_us + fine_scatter_us + download_us;

        let timings = TwoLevelTimings {
            upload_us, coarse_histogram_us, coarse_prefix_us, coarse_copy_us,
            coarse_scatter_us, fine_bucket_us, fine_histogram_us, fine_prefix_us,
            fine_copy_us, fine_scatter_us, download_us, total_us, m,
        };

        (result, timings)
    }
}
