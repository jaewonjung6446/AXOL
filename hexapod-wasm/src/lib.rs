use wasm_bindgen::prelude::*;
use std::f64::consts::TAU;

const WAVE_DIM: usize = 32;
const FEAT_DIM: usize = WAVE_DIM * 2;
const DT: f64 = 0.005;
const DRAG: f64 = 0.98;
const EVAL_STEPS: usize = 200;
const TURN_RATE: f64 = 3.0;

// ─── Simplified Wave Reservoir (WTE Core) ───
struct Reservoir {
    state: [f64; WAVE_DIM],
}

impl Reservoir {
    fn new() -> Self { Self { state: [0.0; WAVE_DIM] } }
    fn reset(&mut self) { self.state = [0.0; WAVE_DIM]; }

    fn feed(&mut self, wave: &[f64; WAVE_DIM]) {
        for i in 0..WAVE_DIM {
            let prev = if i > 0 { self.state[i - 1] } else { self.state[WAVE_DIM - 1] };
            self.state[i] = self.state[i] * 0.55 + wave[i] * 0.45 + prev * 0.12;
        }
    }

    fn features(&self) -> [f64; FEAT_DIM] {
        let mut f = [0.0; FEAT_DIM];
        for i in 0..WAVE_DIM {
            f[i] = self.state[i];
            f[WAVE_DIM + i] = self.state[i] * self.state[i];
        }
        let norm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        for v in &mut f { *v /= norm; }
        f
    }
}

// ─── Gait Pattern ───
#[derive(Clone)]
struct Gait {
    phases: [f64; 6],
    amp: f64,
    freq: f64,
    fitness: f64,
    sel: usize,
    sig: [f64; FEAT_DIM],
}

// ─── WASM Hexapod ───
#[wasm_bindgen]
pub struct HexapodWasm {
    // 2D position + heading
    x: f64,
    z: f64,
    heading: f64,
    vx: f64,
    tilt: f64,
    // Legs
    active: [bool; 6],
    ground: [bool; 6],
    phase: [f64; 6],
    side: [f64; 6],
    // WTE pool
    pool: Vec<Gait>,
    cur: usize,
    res: Reservoir,
    // Evaluation
    step: usize,
    eval_x: f64,
    eval_z: f64,
    eval_ct: usize,
    grow_ct: usize,
    smooth_v: f64,
    // Player input
    input_fwd: f64,
    input_turn: f64,
    // RNG
    rng: u64,
}

impl HexapodWasm {
    fn rand(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % 10000) as f64 / 10000.0
    }

    fn encode_gait(&mut self, ph: &[f64; 6], amp: f64, freq: f64) -> [f64; FEAT_DIM] {
        self.res.reset();
        for i in 0..6 {
            let mut w = [0.0; WAVE_DIM];
            for d in 0..WAVE_DIM {
                let t = d as f64 / WAVE_DIM as f64;
                w[d] = amp * (TAU * (freq * t + ph[i])).sin()
                    + 0.3 * ((i as f64 + 1.0) * TAU * t).cos();
            }
            let n: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
            for v in &mut w { *v /= n; }
            self.res.feed(&w);
        }
        self.res.features()
    }

    fn encode_sensors(&mut self) -> [f64; FEAT_DIM] {
        self.res.reset();
        let mut bw = [0.0; WAVE_DIM];
        for d in 0..WAVE_DIM {
            let t = d as f64 / WAVE_DIM as f64;
            bw[d] = self.vx * (TAU * t).sin() + self.tilt * (TAU * 2.0 * t).cos();
        }
        let n: f64 = bw.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        for v in &mut bw { *v /= n; }
        self.res.feed(&bw);

        for i in 0..6 {
            let mut lw = [0.0; WAVE_DIM];
            let af = if self.active[i] { 1.0 } else { 0.0 };
            let gf = if self.ground[i] { 1.0 } else { -1.0 };
            for d in 0..WAVE_DIM {
                let t = d as f64 / WAVE_DIM as f64;
                lw[d] = af * (TAU * (self.phase[i] + t)).sin()
                    + gf * 0.5 * (TAU * 3.0 * t).cos();
            }
            let n: f64 = lw.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
            for v in &mut lw { *v /= n; }
            self.res.feed(&lw);
        }
        self.res.features()
    }

    fn cosine(a: &[f64; FEAT_DIM], b: &[f64; FEAT_DIM]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
        for i in 0..FEAT_DIM {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let dn = na.sqrt() * nb.sqrt();
        if dn > 1e-10 { d / dn } else { 0.0 }
    }

    fn select_resonance(&mut self) {
        if self.pool.is_empty() { return; }
        if self.rand() < 0.12 {
            self.cur = (self.rand() * self.pool.len() as f64) as usize % self.pool.len();
            return;
        }
        let sig = self.encode_sensors();
        let mut best = 0;
        let mut best_s = f64::NEG_INFINITY;
        for (i, g) in self.pool.iter().enumerate() {
            let s = Self::cosine(&sig, &g.sig) * 0.3 + g.fitness * 5.0;
            if s > best_s { best_s = s; best = i; }
        }
        self.cur = best;
    }

    fn select_best(&mut self) {
        let mut best = 0;
        let mut best_f = f64::NEG_INFINITY;
        for (i, g) in self.pool.iter().enumerate() {
            if g.fitness > best_f { best_f = g.fitness; best = i; }
        }
        self.cur = best;
    }

    fn add_gait(&mut self, phases: [f64; 6], amp: f64, freq: f64) {
        let sig = self.encode_gait(&phases, amp, freq);
        self.pool.push(Gait { phases, amp, freq, fitness: 0.0, sel: 0, sig });
    }

    fn init_pool(&mut self) {
        self.pool.clear();
        self.add_gait([0.0, 0.5, 0.5, 0.0, 0.0, 0.5], 0.4, 2.0);
        self.add_gait([0.0, 0.5, 0.17, 0.67, 0.33, 0.83], 0.35, 1.5);
        self.add_gait([0.0, 0.5, 0.33, 0.83, 0.67, 0.17], 0.38, 1.8);
        self.add_gait([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.3, 2.0);
        for _ in 0..16 {
            let mut ph = [0.0; 6];
            for p in ph.iter_mut() { *p = self.rand(); }
            let a = 0.2 + self.rand() * 0.3;
            let f = 1.0 + self.rand() * 2.0;
            self.add_gait(ph, a, f);
        }
    }

    fn physics_step(&mut self) {
        if self.pool.is_empty() { return; }
        let g = self.pool[self.cur].clone();

        // Turning
        self.heading += self.input_turn * TURN_RATE * DT;

        // Leg phases + thrust
        let (mut grounded, mut thrust) = (0i32, 0.0);
        let (mut ls, mut rs) = (0i32, 0i32);

        for i in 0..6 {
            if !self.active[i] { self.ground[i] = false; continue; }
            self.phase[i] = (self.phase[i] + g.freq * DT) % 1.0;
            let eff = (self.phase[i] + g.phases[i]) % 1.0;
            self.ground[i] = eff < 0.5;
            if self.ground[i] {
                grounded += 1;
                if self.side[i] < 0.0 { ls += 1; } else { rs += 1; }
                thrust += g.amp * g.freq;
            }
        }

        let vert = if grounded >= 3 { 1.0 }
                   else if grounded == 2 { 0.35 }
                   else if grounded == 1 { 0.05 }
                   else { 0.0 };
        let lat = if ls > 0 && rs > 0 { 1.0 }
                  else if ls > 0 || rs > 0 { 0.4 }
                  else { 0.0 };
        self.tilt = (rs as f64 - ls as f64) / 3.0;

        // Forward input modulates thrust
        let fwd = self.input_fwd.clamp(-0.3, 1.0);
        let thrust_mod = if fwd >= 0.0 { fwd } else { fwd * 0.3 };
        self.vx = self.vx * DRAG + thrust * vert * lat * thrust_mod.max(0.02) * DT;
        if self.vx < -0.1 { self.vx = -0.1; }

        // 2D movement in heading direction
        self.x += self.vx * self.heading.cos() * DT;
        self.z += self.vx * self.heading.sin() * DT;
    }

    fn evaluate(&mut self) {
        let dx = self.x - self.eval_x;
        let dz = self.z - self.eval_z;
        let dist = (dx * dx + dz * dz).sqrt();
        self.eval_x = self.x;
        self.eval_z = self.z;
        if self.cur < self.pool.len() {
            let g = &mut self.pool[self.cur];
            g.fitness = g.fitness * 0.6 + dist * 0.4;
            g.sel += 1;
        }
        let speed = dist / (EVAL_STEPS as f64 * DT);
        self.smooth_v = self.smooth_v * 0.5 + speed * 0.5;
        self.select_resonance();
    }

    fn growth_cycle(&mut self) {
        for g in &mut self.pool { g.fitness *= 0.92; }
        if self.pool.len() > 4 {
            self.pool.retain(|g| g.fitness > 0.0005 || g.sel == 0);
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
            let ca = (parent.amp + self.rand() * 0.1 - 0.05).clamp(0.15, 0.6);
            let cf = (parent.freq + self.rand() * 0.3 - 0.15).clamp(0.8, 3.5);
            let cs = self.encode_gait(&cp, ca, cf);
            self.pool.push(Gait {
                phases: cp, amp: ca, freq: cf,
                fitness: parent.fitness * 0.3, sel: 0, sig: cs,
            });
        }

        if self.pool.len() < 20 && self.rand() < 0.3 {
            let mut ph = [0.0; 6];
            for p in ph.iter_mut() { *p = self.rand(); }
            let a = 0.2 + self.rand() * 0.3;
            let f = 1.0 + self.rand() * 2.0;
            let s = self.encode_gait(&ph, a, f);
            self.pool.push(Gait {
                phases: ph, amp: a, freq: f,
                fitness: 0.0, sel: 0, sig: s,
            });
        }

        if self.cur >= self.pool.len() { self.cur = 0; }
    }
}

// ─── WASM API ───
#[wasm_bindgen]
impl HexapodWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut sim = Self {
            x: 0.0, z: 0.0, heading: 0.0,
            vx: 0.0, tilt: 0.0,
            active: [true; 6],
            ground: [true; 6],
            phase: [0.0; 6],
            side: [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            pool: Vec::new(), cur: 0,
            res: Reservoir::new(),
            step: 0, eval_x: 0.0, eval_z: 0.0,
            eval_ct: 0, grow_ct: 0,
            smooth_v: 0.0,
            input_fwd: 1.0, input_turn: 0.0,
            rng: 42,
        };
        sim.init_pool();
        // Initial exploration
        let ps = sim.pool.len();
        for i in 0..ps.min(20) {
            sim.cur = i;
            for _ in 0..EVAL_STEPS {
                sim.physics_step();
                sim.step += 1;
            }
            let dx = sim.x - sim.eval_x;
            let dz = sim.z - sim.eval_z;
            let d = (dx * dx + dz * dz).sqrt();
            sim.eval_x = sim.x;
            sim.eval_z = sim.z;
            if sim.cur < sim.pool.len() {
                let g = &mut sim.pool[sim.cur];
                g.fitness = g.fitness * 0.3 + d * 0.7;
                g.sel += 1;
            }
        }
        sim.select_best();
        sim
    }

    pub fn step_frame(&mut self, steps: usize) {
        for _ in 0..steps {
            self.physics_step();
            self.step += 1;
            self.eval_ct += 1;
            if self.eval_ct >= EVAL_STEPS {
                self.evaluate();
                self.eval_ct = 0;
                self.grow_ct += 1;
                if self.grow_ct >= 3 {
                    self.growth_cycle();
                    self.grow_ct = 0;
                }
            }
        }
    }

    pub fn set_input(&mut self, fwd: f64, turn: f64) {
        self.input_fwd = fwd.clamp(-1.0, 1.0);
        self.input_turn = turn.clamp(-1.0, 1.0);
    }

    pub fn remove_leg(&mut self, idx: usize) {
        if idx < 6 && self.active[idx] {
            self.active[idx] = false;
            for g in &mut self.pool { g.fitness *= 0.05; }
            self.smooth_v = 0.0;
        }
    }

    pub fn restore_leg(&mut self, idx: usize) {
        if idx < 6 {
            self.active[idx] = true;
            for g in &mut self.pool { g.fitness *= 0.1; }
            self.smooth_v = 0.0;
        }
    }

    pub fn restore_all(&mut self) {
        for i in 0..6 { self.active[i] = true; }
        for g in &mut self.pool { g.fitness *= 0.1; }
        self.smooth_v = 0.0;
    }

    pub fn reset(&mut self) { *self = Self::new(); }

    // ── Getters ──
    pub fn speed(&self) -> f64 { self.smooth_v }
    pub fn pool_size(&self) -> usize { self.pool.len() }
    pub fn fitness(&self) -> f64 {
        self.pool.iter().map(|g| g.fitness).fold(0.0, f64::max)
    }
    pub fn pos_x(&self) -> f64 { self.x }
    pub fn pos_z(&self) -> f64 { self.z }
    pub fn get_heading(&self) -> f64 { self.heading }
    pub fn get_tilt(&self) -> f64 { self.tilt }
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }
    pub fn leg_active(&self, i: usize) -> bool { i < 6 && self.active[i] }
    pub fn leg_ground(&self, i: usize) -> bool { i < 6 && self.ground[i] }
    pub fn leg_phase(&self, i: usize) -> f64 {
        if i < 6 { self.phase[i] } else { 0.0 }
    }
    pub fn gait_phase(&self, i: usize) -> f64 {
        if self.cur < self.pool.len() && i < 6 {
            self.pool[self.cur].phases[i]
        } else { 0.0 }
    }
    pub fn gait_amp(&self) -> f64 {
        if self.cur < self.pool.len() { self.pool[self.cur].amp } else { 0.3 }
    }
    pub fn gait_freq(&self) -> f64 {
        if self.cur < self.pool.len() { self.pool[self.cur].freq } else { 1.5 }
    }
    pub fn get_step(&self) -> usize { self.step }
}
