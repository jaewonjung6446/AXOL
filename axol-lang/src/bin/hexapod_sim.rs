//! WTE 6족 보행 시뮬레이션 — 인공 본능 데모
//!
//! cargo run --release --bin hexapod_sim

use axol::wave::Wave;
use axol::types::FloatVec;
use axol::text::reservoir::WaveResonanceReservoir;
use std::time::Instant;

// ─── 물리 상수 ───
const DT: f64 = 0.005;
const EVAL_STEPS: usize = 200;   // 1초/eval
const DRAG: f64 = 0.98;
const WAVE_DIM: usize = 32;

// ─── 프레임 기록 ───
#[derive(Clone)]
struct Frame {
    step: usize,
    x: f64,
    speed: f64,
    phases: [f64; 6],
    amp: f64,
    freq: f64,
    active: [bool; 6],
    pool_size: usize,
    fitness: f64,
    event: String,
}

// ─── 보행 패턴 ───
#[derive(Clone)]
struct GaitPattern {
    phases: [f64; 6],
    amplitude: f64,
    frequency: f64,
    fitness: f64,
    selections: usize,
    wave_sig: Vec<f64>,
}

// ─── 다리 ───
struct Leg {
    phase: f64,
    on_ground: bool,
    active: bool,
    side: f64,
    _pair: usize,
}

// ─── 시뮬레이션 ───
struct HexapodSim {
    x: f64,
    vx: f64,
    tilt: f64,
    legs: Vec<Leg>,
    pool: Vec<GaitPattern>,
    current: usize,
    step: usize,
    eval_x: f64,
    reservoir: WaveResonanceReservoir,
    rng: u64,
    frames: Vec<Frame>,
    smooth_speed: f64,
}

impl HexapodSim {
    fn new() -> Self {
        let legs = vec![
            Leg { phase: 0.0, on_ground: true, active: true, side: -1.0, _pair: 0 },
            Leg { phase: 0.0, on_ground: true, active: true, side:  1.0, _pair: 0 },
            Leg { phase: 0.0, on_ground: true, active: true, side: -1.0, _pair: 1 },
            Leg { phase: 0.0, on_ground: true, active: true, side:  1.0, _pair: 1 },
            Leg { phase: 0.0, on_ground: true, active: true, side: -1.0, _pair: 2 },
            Leg { phase: 0.0, on_ground: true, active: true, side:  1.0, _pair: 2 },
        ];
        let mut sim = HexapodSim {
            x: 0.0, vx: 0.0, tilt: 0.0,
            legs, pool: Vec::new(), current: 0,
            step: 0, eval_x: 0.0,
            reservoir: WaveResonanceReservoir::new(WAVE_DIM),
            rng: 42, frames: Vec::new(), smooth_speed: 0.0,
        };
        sim.init_pool();
        sim
    }

    fn rand(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % 10000) as f64 / 10000.0
    }

    fn init_pool(&mut self) {
        self.pool.clear();
        self.add_gait([0.0, 0.5, 0.5, 0.0, 0.0, 0.5], 0.4, 2.0);   // 삼각
        self.add_gait([0.0, 0.5, 0.17, 0.67, 0.33, 0.83], 0.35, 1.5); // 물결
        self.add_gait([0.0, 0.5, 0.33, 0.83, 0.67, 0.17], 0.38, 1.8); // 대각
        self.add_gait([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.3, 2.0);     // 동기 (나쁨)
        for _ in 0..16 {
            let mut ph = [0.0; 6];
            for p in ph.iter_mut() { *p = self.rand(); }
            let a = 0.2 + self.rand() * 0.3;
            let f = 1.0 + self.rand() * 2.0;
            self.add_gait(ph, a, f);
        }
    }

    fn add_gait(&mut self, phases: [f64; 6], amp: f64, freq: f64) {
        let sig = self.encode_gait(&phases, amp, freq);
        self.pool.push(GaitPattern {
            phases, amplitude: amp, frequency: freq,
            fitness: 0.0, selections: 0, wave_sig: sig,
        });
    }

    fn encode_gait(&mut self, phases: &[f64; 6], amp: f64, freq: f64) -> Vec<f64> {
        let mut waves = Vec::new();
        for (i, &ph) in phases.iter().enumerate() {
            let mut c = vec![0.0f64; WAVE_DIM];
            for d in 0..WAVE_DIM {
                let t = d as f64 / WAVE_DIM as f64;
                c[d] = amp * (std::f64::consts::TAU * (freq * t + ph)).sin()
                    + 0.3 * ((i as f64 + 1.0) * std::f64::consts::TAU * t).cos();
            }
            let n: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
            for v in c.iter_mut() { *v /= n; }
            let f32c: Vec<f32> = c.iter().map(|&v| v as f32).collect();
            waves.push(Wave::from_classical(&FloatVec::new(f32c)));
        }
        self.reservoir.reset();
        self.reservoir.process_sequence(&waves).to_feature_vector()
    }

    fn encode_sensors(&mut self) -> Vec<f64> {
        let mut waves = Vec::new();
        let mut bw = vec![0.0f64; WAVE_DIM];
        for d in 0..WAVE_DIM {
            let t = d as f64 / WAVE_DIM as f64;
            bw[d] = self.vx * (std::f64::consts::TAU * t).sin()
                + self.tilt * (std::f64::consts::TAU * 2.0 * t).cos();
        }
        let n: f64 = bw.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        for v in bw.iter_mut() { *v /= n; }
        let f32bw: Vec<f32> = bw.iter().map(|&v| v as f32).collect();
        waves.push(Wave::from_classical(&FloatVec::new(f32bw)));

        for leg in &self.legs {
            let mut lw = vec![0.0f64; WAVE_DIM];
            let af = if leg.active { 1.0 } else { 0.0 };
            let gf = if leg.on_ground { 1.0 } else { -1.0 };
            for d in 0..WAVE_DIM {
                let t = d as f64 / WAVE_DIM as f64;
                lw[d] = af * (std::f64::consts::TAU * (leg.phase + t)).sin()
                    + gf * 0.5 * (std::f64::consts::TAU * 3.0 * t).cos();
            }
            let n: f64 = lw.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
            for v in lw.iter_mut() { *v /= n; }
            let f32lw: Vec<f32> = lw.iter().map(|&v| v as f32).collect();
            waves.push(Wave::from_classical(&FloatVec::new(f32lw)));
        }
        self.reservoir.reset();
        self.reservoir.process_sequence(&waves).to_feature_vector()
    }

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
        for i in 0..len { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        let dn = na.sqrt() * nb.sqrt();
        if dn > 1e-10 { d / dn } else { 0.0 }
    }

    fn select_resonance(&mut self) {
        if self.pool.is_empty() { return; }
        // ε-greedy
        if self.rand() < 0.12 {
            self.current = (self.rand() * self.pool.len() as f64) as usize % self.pool.len();
            return;
        }
        let sig = self.encode_sensors();
        let mut best = 0;
        let mut best_s = f64::NEG_INFINITY;
        for (i, g) in self.pool.iter().enumerate() {
            let s = Self::cosine(&sig, &g.wave_sig) * 0.3 + g.fitness * 5.0;
            if s > best_s { best_s = s; best = i; }
        }
        self.current = best;
    }

    fn select_best(&mut self) {
        if self.pool.is_empty() { return; }
        let mut best = 0;
        let mut best_f = f64::NEG_INFINITY;
        for (i, g) in self.pool.iter().enumerate() {
            if g.fitness > best_f { best_f = g.fitness; best = i; }
        }
        self.current = best;
    }

    fn tick(&mut self) {
        if self.pool.is_empty() { return; }
        let gait = self.pool[self.current].clone();
        let mut grounded = 0;
        let mut thrust = 0.0;
        let (mut ls, mut rs) = (0, 0);

        for (i, leg) in self.legs.iter_mut().enumerate() {
            if !leg.active { leg.on_ground = false; continue; }
            leg.phase = (leg.phase + gait.frequency * DT) % 1.0;
            let eff = (leg.phase + gait.phases[i]) % 1.0;
            leg.on_ground = eff < 0.5;
            if leg.on_ground {
                grounded += 1;
                if leg.side < 0.0 { ls += 1; } else { rs += 1; }
                thrust += gait.amplitude * gait.frequency;
            }
        }

        let vert = match grounded { g if g >= 3 => 1.0, 2 => 0.35, 1 => 0.05, _ => 0.0 };
        let lat = if ls > 0 && rs > 0 { 1.0 } else if ls > 0 || rs > 0 { 0.4 } else { 0.0 };
        let stability = vert * lat;
        self.tilt = (rs as f64 - ls as f64) / 3.0;
        self.vx = self.vx * DRAG + thrust * stability * DT;
        if self.vx < 0.0 { self.vx *= 0.1; }
        self.x += self.vx * DT;
        self.step += 1;
    }

    fn evaluate(&mut self) -> f64 {
        let dist = self.x - self.eval_x;
        self.eval_x = self.x;
        if self.current < self.pool.len() {
            let g = &mut self.pool[self.current];
            g.fitness = g.fitness * 0.6 + dist * 0.4;
            g.selections += 1;
        }
        let speed = dist / (EVAL_STEPS as f64 * DT);
        self.smooth_speed = self.smooth_speed * 0.5 + speed * 0.5;
        self.select_resonance();
        speed
    }

    fn growth_cycle(&mut self) {
        for g in self.pool.iter_mut() { g.fitness *= 0.92; }
        if self.pool.len() > 4 {
            self.pool.retain(|g| g.fitness > 0.0005 || g.selections == 0);
            if self.pool.is_empty() { self.init_pool(); }
        }
        let best_i = self.pool.iter().enumerate()
            .max_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if self.pool.len() < 25 {
            let parent = self.pool[best_i].clone();
            let mut cp = parent.phases;
            let n = 1 + (self.rand() * 2.0) as usize;
            for _ in 0..n {
                let idx = (self.rand() * 6.0) as usize % 6;
                cp[idx] = (cp[idx] + self.rand() * 0.2 - 0.1).rem_euclid(1.0);
            }
            let ca = (parent.amplitude + self.rand() * 0.1 - 0.05).clamp(0.15, 0.6);
            let cf = (parent.frequency + self.rand() * 0.3 - 0.15).clamp(0.8, 3.5);
            let cs = self.encode_gait(&cp, ca, cf);
            self.pool.push(GaitPattern {
                phases: cp, amplitude: ca, frequency: cf,
                fitness: parent.fitness * 0.3, selections: 0, wave_sig: cs,
            });
        }
        if self.pool.len() < 20 && self.rand() < 0.3 {
            let mut ph = [0.0; 6];
            for p in ph.iter_mut() { *p = self.rand(); }
            let a = 0.2 + self.rand() * 0.3;
            let f = 1.0 + self.rand() * 2.0;
            let s = self.encode_gait(&ph, a, f);
            self.pool.push(GaitPattern {
                phases: ph, amplitude: a, frequency: f,
                fitness: 0.0, selections: 0, wave_sig: s,
            });
        }
        if self.current >= self.pool.len() { self.current = 0; }
    }

    fn remove_leg(&mut self, idx: usize) {
        self.legs[idx].active = false;
        for g in self.pool.iter_mut() { g.fitness *= 0.05; }
        self.smooth_speed = 0.0;
    }

    fn best_fitness(&self) -> f64 {
        self.pool.iter().map(|g| g.fitness).fold(0.0, f64::max)
    }

    fn active_flags(&self) -> [bool; 6] {
        let mut a = [false; 6];
        for (i, l) in self.legs.iter().enumerate() { a[i] = l.active; }
        a
    }

    fn record_frame(&mut self, event: &str) {
        let (ph, am, fr) = if self.current < self.pool.len() {
            let g = &self.pool[self.current];
            (g.phases, g.amplitude, g.frequency)
        } else {
            ([0.0; 6], 0.3, 1.5)
        };
        self.frames.push(Frame {
            step: self.step, x: self.x, speed: self.smooth_speed,
            phases: ph, amp: am, freq: fr,
            active: self.active_flags(),
            pool_size: self.pool.len(), fitness: self.best_fitness(),
            event: event.to_string(),
        });
    }

    fn leg_name(i: usize) -> &'static str {
        match i { 0=>"L1(전좌)", 1=>"R1(전우)", 2=>"L2(중좌)",
                   3=>"R2(중우)", 4=>"L3(후좌)", 5=>"R3(후우)", _=>"?" }
    }
}

// ─── 페이즈 실행 ───
fn run_phase(
    sim: &mut HexapodSim,
    label: &str,
    explore: usize,
    exploit: usize,
    threshold: f64,
) -> f64 {
    println!("  {:>6}  {:>8}  {:>10}  {:>4}  {:>7}  {}",
        "Step", "위치(m)", "속도(평활)", "풀", "적합도", "상태");
    println!("  {}", "─".repeat(62));

    // 라운드로빈 탐색
    let ps = sim.pool.len();
    for i in 0..explore.min(ps) {
        sim.current = i % sim.pool.len();
        let sx = sim.x;
        for _ in 0..EVAL_STEPS { sim.tick(); }
        let d = sim.x - sx;
        if sim.current < sim.pool.len() {
            let g = &mut sim.pool[sim.current];
            g.fitness = g.fitness * 0.3 + d * 0.7;
            g.selections += 1;
        }
        sim.eval_x = sim.x;
        let sp = d / (EVAL_STEPS as f64 * DT);
        sim.smooth_speed = sim.smooth_speed * 0.5 + sp * 0.5;
        sim.record_frame("");
    }
    sim.select_best();

    // 착취 + 성장
    let mut converge_count = 0;
    let mut adapted = false;

    for eval in 0..exploit {
        for _ in 0..EVAL_STEPS { sim.tick(); }
        let _ = sim.evaluate();

        if eval % 3 == 0 { sim.growth_cycle(); }

        // 수렴 판정: 평활 속도가 임계값 이상 3연속
        if sim.smooth_speed > threshold {
            converge_count += 1;
            if converge_count >= 3 && !adapted {
                adapted = true;
                sim.record_frame("adapted");
            }
        } else {
            converge_count = 0;
        }

        let st = if sim.smooth_speed > threshold * 1.2 { "★ 안정" }
                 else if sim.smooth_speed > threshold * 0.6 { "↑ 개선" }
                 else if sim.smooth_speed > threshold * 0.2 { "… 탐색" }
                 else { "✗ 불안정" };

        if eval < 3 || eval % 5 == 0 || (converge_count == 3) {
            println!("  {:>6}  {:>7.2}m  {:>8.3}m/s  {:>4}  {:>7.4}  {}",
                sim.step, sim.x, sim.smooth_speed, sim.pool.len(),
                sim.best_fitness(), st);
        }
        sim.record_frame("");
    }

    println!("  → {} 평활속도: {:.3} m/s", label, sim.smooth_speed);
    sim.smooth_speed
}

// ─── HTML 생성 ───
fn generate_html(frames: &[Frame]) -> String {
    let mut json = String::from("[");
    for (i, f) in frames.iter().enumerate() {
        if i > 0 { json.push(','); }
        json.push_str(&format!(
            "{{\"s\":{},\"x\":{:.2},\"v\":{:.4},\"p\":[{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}],\"a\":{:.3},\"f\":{:.3},\"ac\":[{},{},{},{},{},{}],\"ps\":{},\"fi\":{:.4},\"e\":\"{}\"}}",
            f.step, f.x, f.speed,
            f.phases[0], f.phases[1], f.phases[2], f.phases[3], f.phases[4], f.phases[5],
            f.amp, f.freq,
            f.active[0], f.active[1], f.active[2], f.active[3], f.active[4], f.active[5],
            f.pool_size, f.fitness, f.event
        ));
    }
    json.push(']');

    format!(r##"<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>WTE 6족 보행 — 인공 본능 데모</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#c9d1d9;font-family:'Consolas','D2Coding',monospace;overflow:hidden}}
canvas{{display:block;margin:0 auto}}
.hud{{display:flex;justify-content:center;gap:16px;padding:8px;font-size:13px}}
.m{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 14px}}
.m b{{color:#58a6ff}}
.ctrl{{text-align:center;padding:6px}}
.ctrl button{{background:#21262d;border:1px solid #30363d;color:#c9d1d9;
  padding:4px 14px;border-radius:4px;cursor:pointer;margin:0 3px;font-family:inherit}}
.ctrl button:hover{{background:#30363d}}
.ctrl button.act{{background:#1f6feb;border-color:#1f6feb}}
#timeline{{width:500px;vertical-align:middle;cursor:pointer}}
h1{{text-align:center;font-size:16px;padding:10px;color:#58a6ff}}
.event-banner{{position:absolute;top:80px;left:50%;transform:translateX(-50%);
  background:#da3633;color:#fff;padding:8px 24px;border-radius:8px;font-size:18px;
  font-weight:bold;opacity:0;transition:opacity 0.3s;pointer-events:none}}
.event-banner.show{{opacity:1}}
</style>
</head>
<body>
<h1>WTE 6족 보행 시뮬레이션 — 인공 본능 (Artificial Instinct)</h1>
<canvas id="c"></canvas>
<div class="hud">
 <span class="m" id="mSpeed">속도: <b>0.000</b> m/s</span>
 <span class="m" id="mPool">풀: <b>20</b></span>
 <span class="m" id="mLegs">다리: <b>6</b>/6</span>
 <span class="m" id="mStep">Step: <b>0</b></span>
 <span class="m" id="mFit">적합도: <b>0.000</b></span>
</div>
<div class="ctrl">
 <button id="bPlay" onclick="toggle()">⏸</button>
 <button onclick="ss(1)" id="b1">1x</button>
 <button onclick="ss(2)" id="b2" class="act">2x</button>
 <button onclick="ss(4)" id="b4">4x</button>
 <button onclick="ss(8)" id="b8">8x</button>
 <input type="range" id="timeline" min="0" max="1000" value="0">
</div>
<div class="event-banner" id="banner"></div>

<script>
const F={json};
const cv=document.getElementById('c');
const cx=cv.getContext('2d');
const W=960,H=540;
cv.width=W;cv.height=H;

let fi=0,playing=true,spd=2,lt=0,acc=0,bannerT=0;

// 3D → 2D 사영 (비스듬한 탑다운 뷰)
const CX=W/2,CY=300;
function p3(x,y,z){{return [CX+x*0.9+y*0.35, CY+y*0.55-z*0.85];}}

// 몸체/다리 기하학 — L=좌측(y<0), R=우측(y>0)
const BZ=50;
const HIPS=[[45,-24],[45,24],[0,-26],[0,26],[-45,-24],[-45,24]];
const LABELS=['L1','R1','L2','R2','L3','R3'];

function draw(f,t){{
  cx.clearRect(0,0,W,H);

  // 지면 그리드
  cx.strokeStyle='#1a1f27';cx.lineWidth=1;
  const goff=(f.x*30)%30;
  for(let i=-8;i<16;i++){{
    const gx=i*30-goff;
    const [x1,y1]=p3(gx-120,-180,0);
    const [x2,y2]=p3(gx-120,180,0);
    cx.beginPath();cx.moveTo(x1,y1);cx.lineTo(x2,y2);cx.stroke();
  }}
  for(let j=-6;j<7;j++){{
    const gy=j*30;
    const [x1,y1]=p3(-350,gy,0);
    const [x2,y2]=p3(350,gy,0);
    cx.beginPath();cx.moveTo(x1,y1);cx.lineTo(x2,y2);cx.stroke();
  }}

  // 몸체 그림자 (지면 위)
  cx.fillStyle='rgba(0,0,0,0.25)';
  cx.beginPath();
  const shv=[[60,-24],[60,24],[-60,24],[-60,-24]];
  const [s0x,s0y]=p3(shv[0][0],shv[0][1],0);
  cx.moveTo(s0x,s0y);
  for(let i=1;i<4;i++){{const [sx,sy]=p3(shv[i][0],shv[i][1],0);cx.lineTo(sx,sy);}}
  cx.closePath();cx.fill();

  // 보행 위상
  const phase=(t/1000*f.f)%1.0;

  // 다리 — 뒷다리부터 (깊이 정렬)
  const order=[4,5,2,3,0,1];
  for(const i of order){{
    const hx=HIPS[i][0],hy=HIPS[i][1];
    const dir=hy>0?1:-1;
    const hip=p3(hx,hy,BZ);

    if(!f.ac[i]){{
      const stump=p3(hx,hy+dir*14,BZ-8);
      cx.strokeStyle='#da3633';cx.lineWidth=3;
      cx.beginPath();cx.moveTo(hip[0],hip[1]);cx.lineTo(stump[0],stump[1]);cx.stroke();
      cx.beginPath();
      cx.moveTo(stump[0]-6,stump[1]-6);cx.lineTo(stump[0]+6,stump[1]+6);
      cx.moveTo(stump[0]+6,stump[1]-6);cx.lineTo(stump[0]-6,stump[1]+6);
      cx.stroke();
      cx.fillStyle='#6e1c1c';cx.font='9px monospace';
      cx.fillText(LABELS[i],stump[0]-6,stump[1]-8);
      continue;
    }}

    const eff=(phase+f.p[i])%1.0;
    const stance=eff<0.5;
    const swing=Math.sin(eff*Math.PI*2);

    let fx,fy,fz;
    if(stance){{
      fx=hx+swing*12; fy=hy+dir*55; fz=0;
    }}else{{
      const lift=Math.sin((eff-0.5)*2*Math.PI);
      fx=hx+swing*12; fy=hy+dir*48; fz=Math.max(0,lift*22);
    }}

    const kx=(hx+fx)/2;
    const ky=(hy+fy)/2+dir*10;
    const kz=BZ*0.42+fz*0.3;
    const knee=p3(kx,ky,kz);
    const foot=p3(fx,fy,fz);

    if(!stance&&fz>2){{
      const fs=p3(fx,fy,0);
      cx.fillStyle='rgba(0,0,0,0.12)';
      cx.beginPath();cx.arc(fs[0],fs[1],3,0,Math.PI*2);cx.fill();
    }}

    const lc=stance?'#3fb950':'#d29922';
    cx.strokeStyle=lc;cx.lineWidth=stance?4:3;
    cx.beginPath();cx.moveTo(hip[0],hip[1]);cx.lineTo(knee[0],knee[1]);cx.lineTo(foot[0],foot[1]);cx.stroke();
    cx.fillStyle=lc;
    cx.beginPath();cx.arc(foot[0],foot[1],stance?5:3,0,Math.PI*2);cx.fill();
  }}

  // 몸체 3D 박스 — 오른쪽 측면
  cx.fillStyle='#174ea0';cx.beginPath();
  const rv=[[60,22,BZ+10],[-60,22,BZ+10],[-60,22,BZ-8],[60,22,BZ-8]];
  let [rx,ry]=p3(rv[0][0],rv[0][1],rv[0][2]);cx.moveTo(rx,ry);
  for(let i=1;i<4;i++){{[rx,ry]=p3(rv[i][0],rv[i][1],rv[i][2]);cx.lineTo(rx,ry);}}
  cx.closePath();cx.fill();cx.strokeStyle='#2a6cd4';cx.lineWidth=1;cx.stroke();

  // 정면
  cx.fillStyle='#1a5cc7';cx.beginPath();
  const fv=[[60,-22,BZ+10],[60,22,BZ+10],[60,22,BZ-8],[60,-22,BZ-8]];
  let [ffx,ffy]=p3(fv[0][0],fv[0][1],fv[0][2]);cx.moveTo(ffx,ffy);
  for(let i=1;i<4;i++){{[ffx,ffy]=p3(fv[i][0],fv[i][1],fv[i][2]);cx.lineTo(ffx,ffy);}}
  cx.closePath();cx.fill();cx.strokeStyle='#2a6cd4';cx.lineWidth=1;cx.stroke();

  // 윗면
  cx.fillStyle='#1f6feb';cx.beginPath();
  const ttv=[[60,-22,BZ+10],[60,22,BZ+10],[-60,22,BZ+10],[-60,-22,BZ+10]];
  let [tx,ty]=p3(ttv[0][0],ttv[0][1],ttv[0][2]);cx.moveTo(tx,ty);
  for(let i=1;i<4;i++){{[tx,ty]=p3(ttv[i][0],ttv[i][1],ttv[i][2]);cx.lineTo(tx,ty);}}
  cx.closePath();cx.fill();cx.strokeStyle='#58a6ff';cx.lineWidth=2;cx.stroke();

  // 방향 화살표
  cx.fillStyle='#58a6ff';
  const [a1x,a1y]=p3(72,0,BZ+12);
  const [a2x,a2y]=p3(56,-9,BZ+12);
  const [a3x,a3y]=p3(56,9,BZ+12);
  cx.beginPath();cx.moveTo(a1x,a1y);cx.lineTo(a2x,a2y);cx.lineTo(a3x,a3y);cx.fill();

  // 다리 라벨
  cx.fillStyle='#8b949e';cx.font='10px monospace';
  for(let i=0;i<6;i++){{
    if(!f.ac[i]) continue;
    const lp=p3(HIPS[i][0],HIPS[i][1]*0.5,BZ+18);
    cx.fillText(LABELS[i],lp[0]-7,lp[1]);
  }}

  // 속도 바
  cx.fillStyle='#161b22';cx.fillRect(20,H-50,300,14);
  const speedW=Math.min(f.v/0.8*300,300);
  const barC=f.v>0.3?'#3fb950':f.v>0.1?'#d29922':'#da3633';
  cx.fillStyle=barC;cx.fillRect(20,H-50,Math.max(0,speedW),14);
  cx.fillStyle='#8b949e';cx.font='12px monospace';
  cx.fillText(f.v.toFixed(3)+' m/s',328,H-38);

  cx.fillStyle='#484f58';cx.font='10px monospace';
  cx.fillText('Frame '+fi+'/'+F.length,W-130,H-10);
}}

function updHud(f){{
  document.getElementById('mSpeed').innerHTML='속도: <b>'+f.v.toFixed(3)+'</b> m/s';
  document.getElementById('mPool').innerHTML='풀: <b>'+f.ps+'</b>';
  const nl=f.ac.filter(a=>a).length;
  document.getElementById('mLegs').innerHTML='다리: <b>'+nl+'</b>/6';
  document.getElementById('mStep').innerHTML='Step: <b>'+f.s+'</b>';
  document.getElementById('mFit').innerHTML='적합도: <b>'+f.fi.toFixed(4)+'</b>';
}}

function showBanner(msg){{
  const b=document.getElementById('banner');
  b.textContent=msg;b.classList.add('show');
  bannerT=Date.now();
  setTimeout(()=>b.classList.remove('show'),2000);
}}

function anim(t){{
  if(!lt)lt=t;
  const dt=t-lt;lt=t;

  if(playing){{
    acc+=dt*spd;
    while(acc>150&&fi<F.length-1){{
      acc-=150;
      fi++;
      const f=F[fi];
      if(f.e==='adapted')showBanner('★ 적응 완료!');
      // 다리 손실 감지
      if(fi>0){{
        for(let i=0;i<6;i++){{
          if(F[fi-1].ac[i]&&!f.ac[i]){{
            const names=['L1(전좌)','R1(전우)','L2(중좌)','R2(중우)','L3(후좌)','R3(후우)'];
            showBanner('⚠ 다리 '+names[i]+' 손실!');
          }}
        }}
      }}
      document.getElementById('timeline').value=Math.floor(fi/(F.length-1)*1000);
    }}
  }}

  if(F.length>0){{
    draw(F[fi],t);
    updHud(F[fi]);
  }}
  requestAnimationFrame(anim);
}}

function toggle(){{
  playing=!playing;
  document.getElementById('bPlay').textContent=playing?'⏸':'▶';
}}
function ss(s){{
  spd=s;
  ['b1','b2','b4','b8'].forEach(id=>document.getElementById(id).classList.remove('act'));
  document.getElementById('b'+s).classList.add('act');
}}
document.getElementById('timeline').addEventListener('input',e=>{{
  fi=Math.floor(e.target.value/1000*(F.length-1));
  playing=false;
  document.getElementById('bPlay').textContent='▶';
}});

requestAnimationFrame(anim);
</script>
</body>
</html>"##, json=json)
}

// ─── 메인 ───
fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  WTE 6족 보행 시뮬레이션 — 인공 본능 (Artificial Instinct)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let t_total = Instant::now();
    let mut sim = HexapodSim::new();

    // ── Phase 1: 6족 정상 보행 ──
    println!("[Phase 1] 보행 패턴 자기조직화 (6족 정상)");
    sim.record_frame("start");
    let sp1 = run_phase(&mut sim, "6족", 20, 40, 0.3);
    println!();

    // ── Phase 2: L2 제거 (5족) ──
    let r1 = 2;
    println!("══════════════════════════════════════════════════════════════");
    println!("[Phase 2] ⚠ 다리 {} 제거!", HexapodSim::leg_name(r1));
    println!("══════════════════════════════════════════════════════════════");
    let d1_step = sim.step;
    sim.remove_leg(r1);
    sim.record_frame("damage_L2");
    let sp2 = run_phase(&mut sim, "5족", 15, 30, 0.2);
    let t2 = sim.frames.iter().rev().find(|f| f.event == "adapted" && f.step > d1_step)
        .map(|f| (f.step as f64 - d1_step as f64) * DT);
    println!("  → 적응 시간: {}",
        t2.map_or("미수렴".into(), |t| format!("{:.2}초", t)));
    println!();

    // ── Phase 3: R3 추가 제거 (4족) ──
    let r2 = 5;
    println!("══════════════════════════════════════════════════════════════");
    println!("[Phase 3] ⚠ 다리 {} 추가 제거! (4족)", HexapodSim::leg_name(r2));
    println!("══════════════════════════════════════════════════════════════");
    let d2_step = sim.step;
    sim.remove_leg(r2);
    sim.record_frame("damage_R3");
    let sp3 = run_phase(&mut sim, "4족", 15, 25, 0.15);
    let t3 = sim.frames.iter().rev().find(|f| f.event == "adapted" && f.step > d2_step)
        .map(|f| (f.step as f64 - d2_step as f64) * DT);
    println!("  → 적응 시간: {}",
        t3.map_or("미수렴".into(), |t| format!("{:.2}초", t)));
    println!();

    // ── Phase 4: R1 추가 제거 (3족) ──
    let r3 = 1;
    println!("══════════════════════════════════════════════════════════════");
    println!("[Phase 4] ⚠ 다리 {} 추가 제거! (3족 극한)", HexapodSim::leg_name(r3));
    println!("══════════════════════════════════════════════════════════════");
    let d3_step = sim.step;
    sim.remove_leg(r3);
    sim.record_frame("damage_R1");
    let sp4 = run_phase(&mut sim, "3족", 10, 20, 0.08);
    let t4 = sim.frames.iter().rev().find(|f| f.event == "adapted" && f.step > d3_step)
        .map(|f| (f.step as f64 - d3_step as f64) * DT);
    println!("  → 적응 시간: {}",
        t4.map_or("미수렴".into(), |t| format!("{:.2}초", t)));

    // ── 요약 ──
    let total = t_total.elapsed();
    let mem = sim.pool.len() * (6*8 + 8 + 8 + 8 + 8 +
        sim.pool.first().map_or(0, |g| g.wave_sig.len() * 8));

    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  요약");
    println!("══════════════════════════════════════════════════════════════");
    println!("  {:>14} {:>10} {:>10} {:>6}",  "상태", "속도(m/s)", "적응시간", "다리");
    println!("  {}", "─".repeat(46));
    println!("  {:>14} {:>10.3} {:>10} {:>6}", "6족 정상", sp1, "-", 6);
    println!("  {:>14} {:>10.3} {:>10} {:>6}", "5족 (L2-)", sp2,
        t2.map_or("N/A".into(), |t| format!("{:.2}초", t)), 5);
    println!("  {:>14} {:>10.3} {:>10} {:>6}", "4족 (L2,R3-)", sp3,
        t3.map_or("N/A".into(), |t| format!("{:.2}초", t)), 4);
    println!("  {:>14} {:>10.3} {:>10} {:>6}", "3족 (극한)", sp4,
        t4.map_or("N/A".into(), |t| format!("{:.2}초", t)), 3);
    println!();
    println!("  실행 시간:   {:.1}ms", total.as_secs_f64() * 1000.0);
    println!("  총 스텝:     {}", sim.step);
    println!("  이동 거리:   {:.1}m", sim.x);
    println!("  메모리:      {:.1}KB", mem as f64 / 1024.0);
    println!("  GPU:         없음");
    println!("  재학습:      0회");
    println!("  기록 프레임: {}", sim.frames.len());

    // ── HTML 출력 ──
    let html = generate_html(&sim.frames);
    let out_path = "hexapod_demo.html";
    std::fs::write(out_path, &html).expect("HTML write failed");
    println!();
    println!("  ▶ 시각화: {} ({:.0}KB)", out_path, html.len() as f64 / 1024.0);
    println!();
}
