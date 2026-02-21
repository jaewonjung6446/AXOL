"use strict";

// ═══════════════════════════════════════════════════════
// WGSL Shaders
// ═══════════════════════════════════════════════════════

const COMPUTE_DEPTH_WGSL = /* wgsl */`
struct Params {
  view: mat4x4<f32>,
  n: u32, _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let w = pos[i];
  let viewPos = p.view * vec4(w.xyz, 1.0);
  let depth = -viewPos.z;
  // back-to-front: flip so ascending sort = farthest first
  keys[i] = 0xFFFFFFFFu - bitcast<u32>(max(depth, 0.001));
  vals[i] = i;
}
`;

const AXOL_HISTOGRAM_WGSL = /* wgsl */`
struct Params { n: u32, minKey: u32, invRange: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let bucket = min(u32(f32(keys[i] - p.minKey) * p.invRange), p.n - 1u);
  atomicAdd(&hist[bucket], 1u);
}
`;

const AXOL_SCATTER_WGSL = /* wgsl */`
struct Params { n: u32, minKey: u32, invRange: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> valsIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> valsOut: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let bucket = min(u32(f32(keys[i] - p.minKey) * p.invRange), p.n - 1u);
  let pos = atomicAdd(&offsets[bucket], 1u);
  valsOut[pos] = valsIn[i];
}
`;

const RADIX_HISTOGRAM_WGSL = /* wgsl */`
struct Params { n: u32, shift: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let digit = (keys[i] >> p.shift) & 0xFFu;
  atomicAdd(&hist[digit], 1u);
}
`;

const RADIX_SCATTER_WGSL = /* wgsl */`
struct Params { n: u32, shift: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> keysIn: array<u32>;
@group(0) @binding(2) var<storage, read> valsIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> keysOut: array<u32>;
@group(0) @binding(5) var<storage, read_write> valsOut: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let key = keysIn[i];
  let digit = (key >> p.shift) & 0xFFu;
  let pos = atomicAdd(&offsets[digit], 1u);
  keysOut[pos] = key;
  valsOut[pos] = valsIn[i];
}
`;

// Blelloch exclusive prefix sum — 256 threads, 512 elements per workgroup
const PREFIX_SCAN_WGSL = /* wgsl */`
struct Params { n: u32, _p1: u32, _p2: u32, _p3: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> sh: array<u32, 512>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let t = lid.x;
  let base = wid.x * 512u;
  let i0 = base + t;
  let i1 = base + t + 256u;
  sh[t]       = select(0u, data[i0], i0 < p.n);
  sh[t + 256u] = select(0u, data[i1], i1 < p.n);

  // up-sweep
  var off = 1u;
  for (var d = 256u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (t < d) {
      let ai = off * (2u * t + 1u) - 1u;
      let bi = off * (2u * t + 2u) - 1u;
      sh[bi] += sh[ai];
    }
    off <<= 1u;
  }

  if (t == 0u) {
    blockSums[wid.x] = sh[511u];
    sh[511u] = 0u;
  }

  // down-sweep
  for (var d = 1u; d < 512u; d <<= 1u) {
    off >>= 1u;
    workgroupBarrier();
    if (t < d) {
      let ai = off * (2u * t + 1u) - 1u;
      let bi = off * (2u * t + 2u) - 1u;
      let tmp = sh[ai];
      sh[ai] = sh[bi];
      sh[bi] += tmp;
    }
  }
  workgroupBarrier();

  if (i0 < p.n) { data[i0] = sh[t]; }
  if (i1 < p.n) { data[i1] = sh[t + 256u]; }
}
`;

const PREFIX_PROPAGATE_WGSL = /* wgsl */`
struct Params { n: u32, _p1: u32, _p2: u32, _p3: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  if (wid.x == 0u) { return; }
  let prefix = blockSums[wid.x];
  let i0 = wid.x * 512u + lid.x;
  let i1 = i0 + 256u;
  if (i0 < p.n) { data[i0] += prefix; }
  if (i1 < p.n) { data[i1] += prefix; }
}
`;

const RENDER_WGSL = /* wgsl */`
struct Camera {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  n: u32, _p1: u32, _p2: u32, _p3: u32,
}
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
  let quadUV = array<vec2<f32>, 6>(
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0),
    vec2(-1.0,-1.0), vec2(1.0,1.0),  vec2(-1.0,1.0)
  );
  let uv = quadUV[vid];
  let idx = sortedIdx[iid];
  let worldPos = positions[idx].xyz;
  let scale = scales[idx];

  var vp = cam.view * vec4(worldPos, 1.0);
  vp = vec4(vp.x + uv.x * scale, vp.y + uv.y * scale, vp.z, vp.w);
  let cp = cam.proj * vp;

  var out: VSOut;
  out.pos = cp;
  out.uv = uv;
  out.color = colors[idx];
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let gauss = exp(-4.0 * r2);
  let a = in.color.a * gauss;
  return vec4(in.color.rgb * a, a);
}
`;

// ═══════════════════════════════════════════════════════
// Math Utilities
// ═══════════════════════════════════════════════════════

function mat4Perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)*nf, -1,
    0, 0, 2*far*near*nf, 0
  ]);
}

// Column-major 4x4 lookAt matrix
function lookAt(eye, center, up) {
  const f = [center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]];
  let l = Math.hypot(...f); f[0]/=l; f[1]/=l; f[2]/=l;
  const s = [f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0]];
  l = Math.hypot(...s); s[0]/=l; s[1]/=l; s[2]/=l;
  const u = [s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]];
  return new Float32Array([
    s[0], u[0], -f[0], 0,
    s[1], u[1], -f[1], 0,
    s[2], u[2], -f[2], 0,
    -(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]),
    -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]),
    (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]),
    1
  ]);
}

// ═══════════════════════════════════════════════════════
// Scene Generation
// ═══════════════════════════════════════════════════════

function generateScene(N) {
  const pos = new Float32Array(N * 4);
  const col = new Float32Array(N * 4);
  const scl = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const r = Math.random();
    let x, y, z, cr, cg, cb, opacity, s;

    if (r < 0.55) {
      // spiral arm
      const arm = Math.floor(Math.random() * 3);
      const dist = Math.pow(Math.random(), 0.5) * 5;
      const angle = arm * (2*Math.PI/3) + dist*0.6 + (Math.random()-0.5)*0.4;
      x = Math.cos(angle)*dist + (Math.random()-0.5)*0.3;
      z = Math.sin(angle)*dist + (Math.random()-0.5)*0.3;
      y = (Math.random()-0.5)*0.3*(1 - dist/6);
      const t = dist / 5;
      cr = 0.3+t*0.5; cg = 0.4+(1-t)*0.4; cb = 0.7+(1-t)*0.3;
      opacity = 0.25 + Math.random()*0.35;
      s = 0.04 + Math.random()*0.06;
    } else if (r < 0.8) {
      // core
      const dist = Math.pow(Math.random(),2) * 1.5;
      const th = Math.random()*Math.PI*2;
      const ph = (Math.random()-0.5)*Math.PI;
      x = Math.cos(th)*Math.cos(ph)*dist;
      y = Math.sin(ph)*dist*0.4;
      z = Math.sin(th)*Math.cos(ph)*dist;
      cr=1.0; cg=0.85-dist*0.2; cb=0.3+dist*0.3;
      opacity = 0.4 + Math.random()*0.4;
      s = 0.03 + Math.random()*0.05;
    } else {
      // halo
      const dist = 1 + Math.random()*7;
      const th = Math.random()*Math.PI*2;
      const ph = (Math.random()-0.5)*Math.PI*0.4;
      x = Math.cos(th)*Math.cos(ph)*dist;
      y = Math.sin(ph)*dist*0.25;
      z = Math.sin(th)*Math.cos(ph)*dist;
      cr=0.7+Math.random()*0.3; cg=0.7+Math.random()*0.3; cb=0.85+Math.random()*0.15;
      opacity = 0.15 + Math.random()*0.2;
      s = 0.015 + Math.random()*0.025;
    }

    pos[i*4]=x; pos[i*4+1]=y; pos[i*4+2]=z; pos[i*4+3]=1;
    col[i*4]=cr; col[i*4+1]=cg; col[i*4+2]=cb; col[i*4+3]=opacity;
    scl[i]=s;
  }
  return { positions: pos, colors: col, scales: scl };
}

// ═══════════════════════════════════════════════════════
// Prefix Sum Helper
// ═══════════════════════════════════════════════════════

const SCAN_BLOCK = 512; // 256 threads × 2 elements

class PrefixSum {
  constructor(device, scanPipeline, propPipeline, bgl) {
    this.device = device;
    this.scanPL = scanPipeline;
    this.propPL = propPipeline;
    this.bgl = bgl;
  }

  // Prepare buffers and bind groups for a specific size
  prepare(dataBuffer, maxN) {
    const levels = [];
    let s = maxN;
    while (s > SCAN_BLOCK) {
      const nb = Math.ceil(s / SCAN_BLOCK);
      levels.push({ size: s, numBlocks: nb });
      s = nb;
    }
    levels.push({ size: s, numBlocks: 1 });

    // Create block sum buffers + param uniforms + bind groups
    const blockBufs = [];
    for (let i = 0; i < levels.length; i++) {
      const nb = (i < levels.length - 1) ? levels[i].numBlocks : 1;
      const padded = Math.max(nb, 4); // minimum size
      blockBufs.push(this.device.createBuffer({
        size: padded * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
    }

    const scanBGs = [];
    const propBGs = [];

    for (let i = 0; i < levels.length; i++) {
      const dataBuf = (i === 0) ? dataBuffer : blockBufs[i - 1];
      const sumBuf = blockBufs[i];
      const paramBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([levels[i].size, 0, 0, 0]));

      scanBGs.push(this.device.createBindGroup({
        layout: this.bgl,
        entries: [
          { binding: 0, resource: { buffer: dataBuf } },
          { binding: 1, resource: { buffer: sumBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ],
      }));
    }

    for (let i = 0; i < levels.length - 1; i++) {
      const dataBuf = (i === 0) ? dataBuffer : blockBufs[i - 1];
      const sumBuf = blockBufs[i];
      const paramBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, new Uint32Array([levels[i].size, 0, 0, 0]));

      propBGs.push(this.device.createBindGroup({
        layout: this.bgl,
        entries: [
          { binding: 0, resource: { buffer: dataBuf } },
          { binding: 1, resource: { buffer: sumBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ],
      }));
    }

    return { levels, blockBufs, scanBGs, propBGs };
  }

  encode(pass, prep) {
    const { levels, scanBGs, propBGs } = prep;
    // Scan bottom-up
    for (let i = 0; i < levels.length; i++) {
      pass.setPipeline(this.scanPL);
      pass.setBindGroup(0, scanBGs[i]);
      pass.dispatchWorkgroups(Math.max(1, levels[i].numBlocks));
    }
    // Propagate top-down
    for (let i = levels.length - 2; i >= 0; i--) {
      pass.setPipeline(this.propPL);
      pass.setBindGroup(0, propBGs[i]);
      pass.dispatchWorkgroups(levels[i].numBlocks);
    }
  }
}

// ═══════════════════════════════════════════════════════
// Main Application
// ═══════════════════════════════════════════════════════

class App {
  constructor() {
    this.sortMode = 'axol'; // 'none' | 'axol' | 'radix'
    this.numSplats = 200000;
    this.camDist = 12;
    this.camTheta = 0;
    this.camPhi = 0.35;
    this.dragging = false;
    this.fpsSmooth = 60;
    this.frameTimeSmooth = 16;
  }

  async init() {
    const canvas = document.getElementById('canvas');
    if (!navigator.gpu) {
      document.getElementById('fallback').style.display = 'flex';
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      document.getElementById('fallback').style.display = 'flex';
      return false;
    }

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxBufferSize: 256 * 1024 * 1024,
      }
    });

    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });

    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    this.aspect = canvas.width / canvas.height;

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth * devicePixelRatio;
      canvas.height = window.innerHeight * devicePixelRatio;
      this.aspect = canvas.width / canvas.height;
      this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });
    });

    this.setupInput(canvas);
    this.createPipelines();
    this.loadScene();
    this.setupUI();
    return true;
  }

  setupInput(canvas) {
    canvas.addEventListener('mousedown', (e) => { this.dragging = true; this.lastX = e.clientX; this.lastY = e.clientY; });
    canvas.addEventListener('mouseup', () => { this.dragging = false; });
    canvas.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      this.camTheta -= (e.clientX - this.lastX) * 0.005;
      this.camPhi = Math.max(-1.2, Math.min(1.2, this.camPhi + (e.clientY - this.lastY) * 0.005));
      this.lastX = e.clientX; this.lastY = e.clientY;
    });
    canvas.addEventListener('wheel', (e) => {
      this.camDist = Math.max(3, Math.min(30, this.camDist + e.deltaY * 0.01));
    });
  }

  setupUI() {
    document.querySelectorAll('#sortBtns .btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#sortBtns .btn').forEach(b => b.className = 'btn');
        const mode = btn.dataset.sort;
        btn.classList.add(mode === 'none' ? 'active-none' : mode === 'radix' ? 'active-radix' : 'active');
        this.sortMode = mode;
      });
    });
    document.querySelectorAll('#countBtns .btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#countBtns .btn').forEach(b => b.className = 'btn');
        btn.classList.add('active');
        this.numSplats = parseInt(btn.dataset.count);
        this.loadScene();
      });
    });
  }

  createPipelines() {
    const dev = this.device;

    // Helper to create compute pipeline
    const mkCompute = (code, bgls) => {
      const sm = dev.createShaderModule({ code });
      const layout = dev.createPipelineLayout({ bindGroupLayouts: bgls });
      return dev.createComputePipeline({ layout, compute: { module: sm, entryPoint: 'main' } });
    };

    // --- Bind group layouts ---

    // depth: uniform, storage-r, storage-rw, storage-rw
    this.depthBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });

    // axol hist: uniform, storage-r, storage-rw(atomic)
    this.axolHistBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });

    // axol scatter: uniform, storage-r, storage-r, storage-rw(atomic), storage-rw
    this.axolScatBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });

    // radix hist: uniform, storage-r, storage-rw(atomic)
    this.radixHistBGL = this.axolHistBGL; // same layout

    // radix scatter: uniform, storage-r, storage-r, storage-rw(atomic), storage-rw, storage-rw
    this.radixScatBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });

    // prefix sum: storage-rw, storage-rw, uniform
    this.prefixBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ] });

    // render: uniform, storage-r ×4, storage-r
    this.renderBGL = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ] });

    // --- Pipelines ---
    this.depthPL = mkCompute(COMPUTE_DEPTH_WGSL, [this.depthBGL]);
    this.axolHistPL = mkCompute(AXOL_HISTOGRAM_WGSL, [this.axolHistBGL]);
    this.axolScatPL = mkCompute(AXOL_SCATTER_WGSL, [this.axolScatBGL]);
    this.radixHistPL = mkCompute(RADIX_HISTOGRAM_WGSL, [this.radixHistBGL]);
    this.radixScatPL = mkCompute(RADIX_SCATTER_WGSL, [this.radixScatBGL]);
    this.prefixScanPL = mkCompute(PREFIX_SCAN_WGSL, [this.prefixBGL]);
    this.prefixPropPL = mkCompute(PREFIX_PROPAGATE_WGSL, [this.prefixBGL]);

    // Render pipeline
    const renderSM = dev.createShaderModule({ code: RENDER_WGSL });
    this.renderPL = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.renderBGL] }),
      vertex: { module: renderSM, entryPoint: 'vs' },
      fragment: {
        module: renderSM, entryPoint: 'fs',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.prefixSum = new PrefixSum(dev, this.prefixScanPL, this.prefixPropPL, this.prefixBGL);
  }

  loadScene() {
    const dev = this.device;
    const N = this.numSplats;
    const scene = generateScene(N);

    const mkBuf = (data, usage) => {
      const buf = dev.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(buf, 0, data);
      return buf;
    };
    const mkEmpty = (bytes, usage) => dev.createBuffer({ size: bytes, usage });

    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;
    const CS = GPUBufferUsage.COPY_SRC;

    this.posBuf = mkBuf(scene.positions, S);
    this.colBuf = mkBuf(scene.colors, S);
    this.sclBuf = mkBuf(scene.scales, S);

    this.keysA = mkEmpty(N * 4, S | CD);
    this.keysB = mkEmpty(N * 4, S | CD);
    this.valsA = mkEmpty(N * 4, S | CD);
    this.valsB = mkEmpty(N * 4, S | CD);

    // AXOL buffers
    this.axolHist = mkEmpty(N * 4, S | CD | CS);
    this.axolOffsets = mkEmpty(N * 4, S | CD);

    // Radix buffers
    this.radixHist = mkEmpty(256 * 4, S | CD | CS);
    this.radixOffsets = mkEmpty(256 * 4, S | CD);

    // Identity indices (for no-sort mode)
    const identity = new Uint32Array(N);
    for (let i = 0; i < N; i++) identity[i] = i;
    this.identityBuf = mkBuf(identity, S);

    // Camera uniform (view 64B + proj 64B + n 4B + pad 12B = 144B)
    this.camBuf = mkEmpty(144, GPUBufferUsage.UNIFORM | CD);

    // Depth params (viewRow2 16B × 4 rows? no, just row2 16B + n 16B = 32B... let me use 80B)
    // Params: viewRow0-3 (64B) + n,pad,pad,pad (16B) = 80B
    this.depthParamBuf = mkEmpty(80, GPUBufferUsage.UNIFORM | CD);

    // AXOL sort params: n, minKey, invRange, pad = 16B
    this.axolParamBuf = mkEmpty(16, GPUBufferUsage.UNIFORM | CD);

    // Prefix sum prep for AXOL (N buckets)
    this.axolPrefixPrep = this.prefixSum.prepare(this.axolHist, N);

    // Prefix sum prep for radix (256 buckets) - single workgroup
    this.radixPrefixPrep = this.prefixSum.prepare(this.radixHist, 256);

    // Create bind groups
    this.createBindGroups();
  }

  createBindGroups() {
    const dev = this.device;
    const N = this.numSplats;

    // Depth compute
    this.depthBG = dev.createBindGroup({
      layout: this.depthBGL,
      entries: [
        { binding: 0, resource: { buffer: this.depthParamBuf } },
        { binding: 1, resource: { buffer: this.posBuf } },
        { binding: 2, resource: { buffer: this.keysA } },
        { binding: 3, resource: { buffer: this.valsA } },
      ],
    });

    // AXOL histogram
    this.axolHistBG = dev.createBindGroup({
      layout: this.axolHistBGL,
      entries: [
        { binding: 0, resource: { buffer: this.axolParamBuf } },
        { binding: 1, resource: { buffer: this.keysA } },
        { binding: 2, resource: { buffer: this.axolHist } },
      ],
    });

    // AXOL scatter → output to valsB
    this.axolScatBG = dev.createBindGroup({
      layout: this.axolScatBGL,
      entries: [
        { binding: 0, resource: { buffer: this.axolParamBuf } },
        { binding: 1, resource: { buffer: this.keysA } },
        { binding: 2, resource: { buffer: this.valsA } },
        { binding: 3, resource: { buffer: this.axolOffsets } },
        { binding: 4, resource: { buffer: this.valsB } },
      ],
    });

    // Radix: per-pass bind groups with pre-baked shift params
    // Pass 0: shift=0, A→B; Pass 1: shift=8, B→A; Pass 2: shift=16, A→B; Pass 3: shift=24, B→A
    const keysBufs = [this.keysA, this.keysB];
    const valsBufs = [this.valsA, this.valsB];
    this.radixHistBGs_perPass = [];
    this.radixScatBGs_perPass = [];
    for (let pass = 0; pass < 4; pass++) {
      const shift = pass * 8;
      const src = pass % 2;
      const dst = 1 - src;
      const paramBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(paramBuf, 0, new Uint32Array([N, shift, 0, 0]));

      this.radixHistBGs_perPass.push(dev.createBindGroup({ layout: this.radixHistBGL, entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: keysBufs[src] } },
        { binding: 2, resource: { buffer: this.radixHist } },
      ] }));
      this.radixScatBGs_perPass.push(dev.createBindGroup({ layout: this.radixScatBGL, entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: keysBufs[src] } },
        { binding: 2, resource: { buffer: valsBufs[src] } },
        { binding: 3, resource: { buffer: this.radixOffsets } },
        { binding: 4, resource: { buffer: keysBufs[dst] } },
        { binding: 5, resource: { buffer: valsBufs[dst] } },
      ] }));
    }

    // Render bind groups — AXOL uses valsB, Radix uses valsA (after 4 passes), None uses identity
    this.renderBG_axol = dev.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: this.valsB } },
        { binding: 2, resource: { buffer: this.posBuf } },
        { binding: 3, resource: { buffer: this.colBuf } },
        { binding: 4, resource: { buffer: this.sclBuf } },
      ],
    });
    this.renderBG_radix = dev.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: this.valsA } },
        { binding: 2, resource: { buffer: this.posBuf } },
        { binding: 3, resource: { buffer: this.colBuf } },
        { binding: 4, resource: { buffer: this.sclBuf } },
      ],
    });
    this.renderBG_none = dev.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: this.identityBuf } },
        { binding: 2, resource: { buffer: this.posBuf } },
        { binding: 3, resource: { buffer: this.colBuf } },
        { binding: 4, resource: { buffer: this.sclBuf } },
      ],
    });
  }

  frame() {
    const N = this.numSplats;
    const wg = Math.ceil(N / 256);

    // Update camera
    if (!this.dragging) this.camTheta += 0.003;
    const eye = [
      this.camDist * Math.cos(this.camPhi) * Math.sin(this.camTheta),
      this.camDist * Math.sin(this.camPhi),
      this.camDist * Math.cos(this.camPhi) * Math.cos(this.camTheta),
    ];
    const view = lookAt(eye, [0,0,0], [0,1,0]);
    const proj = mat4Perspective(Math.PI/4, this.aspect, 0.1, 100);

    // Upload camera
    const camData = new Float32Array(36); // 16+16+4 = 36 floats = 144 bytes
    camData.set(view, 0);
    camData.set(proj, 16);
    const camUints = new Uint32Array(camData.buffer);
    camUints[32] = N;
    this.device.queue.writeBuffer(this.camBuf, 0, camData);

    // Upload depth params (view matrix rows + n)
    const depthParams = new Float32Array(20); // 16 floats (view) + 4 (n,pad)
    depthParams.set(view, 0);
    const dpUints = new Uint32Array(depthParams.buffer);
    dpUints[16] = N;
    this.device.queue.writeBuffer(this.depthParamBuf, 0, depthParams);

    // AXOL params: compute min/max key from camera bounds
    // key = 0xFFFFFFFF - bitcast<u32>(depth), so larger depth → smaller key
    const minDepth = Math.max(0.1, this.camDist - 8);
    const maxDepth = this.camDist + 8;
    const tmpF = new Float32Array([maxDepth, minDepth]);
    const tmpU = new Uint32Array(tmpF.buffer);
    // minKey = key(maxDepth) = 0xFFFFFFFF - bits(maxDepth), farthest splat
    // maxKey = key(minDepth) = 0xFFFFFFFF - bits(minDepth), nearest splat
    const minKey = (0xFFFFFFFF - tmpU[0]) >>> 0;
    const maxKey = (0xFFFFFFFF - tmpU[1]) >>> 0;
    const range = (maxKey - minKey) >>> 0;
    const invRange = range > 0 ? N / range : 1.0;

    const axolP = new ArrayBuffer(16);
    new Uint32Array(axolP)[0] = N;
    new Uint32Array(axolP)[1] = minKey;
    new Float32Array(axolP)[2] = invRange;
    this.device.queue.writeBuffer(this.axolParamBuf, 0, axolP);

    const encoder = this.device.createCommandEncoder();

    // ── Compute depth keys ──
    if (this.sortMode !== 'none') {
      const cp = encoder.beginComputePass();
      cp.setPipeline(this.depthPL);
      cp.setBindGroup(0, this.depthBG);
      cp.dispatchWorkgroups(wg);
      cp.end();
    }

    // ── Sort ──
    if (this.sortMode === 'axol') {
      // Clear histogram
      encoder.clearBuffer(this.axolHist, 0, N * 4);

      const cp = encoder.beginComputePass();

      // Histogram
      cp.setPipeline(this.axolHistPL);
      cp.setBindGroup(0, this.axolHistBG);
      cp.dispatchWorkgroups(wg);

      // Prefix sum
      this.prefixSum.encode(cp, this.axolPrefixPrep);

      cp.end();

      // Copy histogram → offsets
      encoder.copyBufferToBuffer(this.axolHist, 0, this.axolOffsets, 0, N * 4);

      // Scatter
      const cp2 = encoder.beginComputePass();
      cp2.setPipeline(this.axolScatPL);
      cp2.setBindGroup(0, this.axolScatBG);
      cp2.dispatchWorkgroups(wg);
      cp2.end();

    } else if (this.sortMode === 'radix') {
      // 4-pass LSD radix sort — use pre-created param buffers per pass
      for (let pass = 0; pass < 4; pass++) {
        const src = pass % 2; // 0=A→B, 1=B→A

        // Clear radix histogram
        encoder.clearBuffer(this.radixHist, 0, 256 * 4);

        const cp = encoder.beginComputePass();

        // Histogram
        cp.setPipeline(this.radixHistPL);
        cp.setBindGroup(0, this.radixHistBGs_perPass[pass]);
        cp.dispatchWorkgroups(wg);

        // Prefix sum (256 elements = single workgroup)
        this.prefixSum.encode(cp, this.radixPrefixPrep);

        cp.end();

        // Copy hist → offsets
        encoder.copyBufferToBuffer(this.radixHist, 0, this.radixOffsets, 0, 256 * 4);

        // Scatter
        const cp2 = encoder.beginComputePass();
        cp2.setPipeline(this.radixScatPL);
        cp2.setBindGroup(0, this.radixScatBGs_perPass[pass]);
        cp2.dispatchWorkgroups(wg);
        cp2.end();
      }
      // After 4 passes (even), result is back in keysA/valsA
    }

    // ── Render ──
    const tex = this.ctx.getCurrentTexture();
    const rp = encoder.beginRenderPass({
      colorAttachments: [{
        view: tex.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1 },
      }],
    });

    rp.setPipeline(this.renderPL);
    const bg = this.sortMode === 'axol' ? this.renderBG_axol
             : this.sortMode === 'radix' ? this.renderBG_radix
             : this.renderBG_none;
    rp.setBindGroup(0, bg);
    rp.draw(6, N);
    rp.end();

    this.device.queue.submit([encoder.finish()]);
  }

  run() {
    let lastTime = performance.now();
    const loop = () => {
      const now = performance.now();
      const dt = now - lastTime;
      lastTime = now;

      this.frame();

      // FPS
      this.fpsSmooth = this.fpsSmooth * 0.92 + (1000/dt) * 0.08;
      this.frameTimeSmooth = this.frameTimeSmooth * 0.92 + dt * 0.08;

      document.getElementById('fps').textContent = Math.round(this.fpsSmooth);
      document.getElementById('frameTime').textContent = this.frameTimeSmooth.toFixed(1) + ' ms';

      const q = document.getElementById('quality');
      if (this.sortMode === 'none') {
        q.textContent = 'No sorting — notice incorrect blending artifacts';
        q.style.color = '#E74C3C';
      } else if (this.sortMode === 'axol') {
        q.textContent = 'AXOL scatter: O(n) single-pass, ~63% exact match but 99.999% rank correlation';
        q.style.color = '#F0AD4E';
      } else {
        q.textContent = 'Radix sort: 4-pass exact sort, 100% accuracy';
        q.style.color = '#999';
      }

      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}

// ═══════════════════════════════════════════════════════
// Entry
// ═══════════════════════════════════════════════════════

window.addEventListener('load', async () => {
  const app = new App();
  const ok = await app.init();
  if (ok) app.run();
});
