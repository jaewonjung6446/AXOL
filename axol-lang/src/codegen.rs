//! AXOL Code Generator — produces valid .axol programs from algorithm specifications.
//!
//! This module provides two layers:
//! 1. `AxolGenerator` — low-level builder for composing arbitrary .axol source
//! 2. Predefined algorithm templates — high-level functions that produce complete programs

use std::fmt::Write;

// ---------------------------------------------------------------------------
// Low-level builder
// ---------------------------------------------------------------------------

/// Fluent builder for .axol source code.
pub struct AxolGenerator {
    lines: Vec<String>,
}

impl AxolGenerator {
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    /// Add a comment line.
    pub fn comment(&mut self, text: &str) -> &mut Self {
        self.lines.push(format!("# {}", text));
        self
    }

    /// Add a blank line.
    pub fn blank(&mut self) -> &mut Self {
        self.lines.push(String::new());
        self
    }

    /// Emit a `declare` block.
    pub fn declare(
        &mut self,
        name: &str,
        inputs: &[(&str, usize)],
        outputs: &[&str],
        relations: &[(&str, &[&str], &str)], // (target, sources, relation_kind)
        quality: Option<(f64, f64)>,
    ) -> &mut Self {
        self.lines.push(format!("declare \"{}\" {{", name));
        for (inp_name, dim) in inputs {
            self.lines.push(format!("    input {}({})", inp_name, dim));
        }
        for out in outputs {
            self.lines.push(format!("    output {}", out));
        }
        for (target, sources, kind) in relations {
            let src_str = sources.join(", ");
            self.lines.push(format!("    relate {} <- {} via {}", target, src_str, kind));
        }
        if let Some((omega, phi)) = quality {
            self.lines.push(format!("    quality omega={} phi={}", omega, phi));
        }
        self.lines.push("}".to_string());
        self
    }

    /// Emit a `weave` command.
    pub fn weave(&mut self, name: &str, quantum: bool, seed: u64) -> &mut Self {
        self.lines.push(format!(
            "weave {} quantum={} seed={}",
            name,
            if quantum { 1 } else { 0 },
            seed
        ));
        self
    }

    /// Emit an `observe` command.
    pub fn observe(&mut self, name: &str, inputs: &[(&str, &[f64])]) -> &mut Self {
        let mut s = format!("observe {} {{", name);
        let parts: Vec<String> = inputs
            .iter()
            .map(|(k, v)| {
                let vals: Vec<String> = v.iter().map(|x| format_float(*x)).collect();
                format!(" {} = [{}]", k, vals.join(", "))
            })
            .collect();
        s.push_str(&parts.join(","));
        s.push_str(" }");
        self.lines.push(s);
        self
    }

    /// Emit a `reobserve` command.
    pub fn reobserve(
        &mut self,
        name: &str,
        count: usize,
        inputs: &[(&str, &[f64])],
    ) -> &mut Self {
        let mut s = format!("reobserve {} count={} {{", name, count);
        let parts: Vec<String> = inputs
            .iter()
            .map(|(k, v)| {
                let vals: Vec<String> = v.iter().map(|x| format_float(*x)).collect();
                format!(" {} = [{}]", k, vals.join(", "))
            })
            .collect();
        s.push_str(&parts.join(","));
        s.push_str(" }");
        self.lines.push(s);
        self
    }

    /// Emit a `gate` command.
    pub fn gate(&mut self, gate_type: &str, inputs: &[(&str, &[f64])]) -> &mut Self {
        let mut s = format!("gate {} {{", gate_type);
        let parts: Vec<String> = inputs
            .iter()
            .map(|(k, v)| {
                let vals: Vec<String> = v.iter().map(|x| format_float(*x)).collect();
                format!(" {} = [{}]", k, vals.join(", "))
            })
            .collect();
        s.push_str(&parts.join(","));
        s.push_str(" }");
        self.lines.push(s);
        self
    }

    /// Emit a `compose` command.
    pub fn compose(&mut self, name: &str, stages: &[&str]) -> &mut Self {
        self.lines.push(format!(
            "compose \"{}\" stages=[{}]",
            name,
            stages.join(", ")
        ));
        self
    }

    /// Emit a `confident` command.
    pub fn confident(
        &mut self,
        name: &str,
        max_obs: usize,
        threshold: f64,
        inputs: &[(&str, &[f64])],
    ) -> &mut Self {
        let mut s = format!(
            "confident {} max={} threshold={} {{",
            name, max_obs, threshold
        );
        let parts: Vec<String> = inputs
            .iter()
            .map(|(k, v)| {
                let vals: Vec<String> = v.iter().map(|x| format_float(*x)).collect();
                format!(" {} = [{}]", k, vals.join(", "))
            })
            .collect();
        s.push_str(&parts.join(","));
        s.push_str(" }");
        self.lines.push(s);
        self
    }

    /// Emit an `iterate` command.
    pub fn iterate(
        &mut self,
        name: &str,
        max_iter: usize,
        converge_type: &str,
        converge_value: f64,
        inputs: &[(&str, &[f64])],
    ) -> &mut Self {
        let mut s = format!(
            "iterate {} max={} converge={} value={} {{",
            name, max_iter, converge_type, converge_value
        );
        let parts: Vec<String> = inputs
            .iter()
            .map(|(k, v)| {
                let vals: Vec<String> = v.iter().map(|x| format_float(*x)).collect();
                format!(" {} = [{}]", k, vals.join(", "))
            })
            .collect();
        s.push_str(&parts.join(","));
        s.push_str(" }");
        self.lines.push(s);
        self
    }

    /// Emit a `design` command.
    pub fn design(
        &mut self,
        name: &str,
        dim: usize,
        n_basins: usize,
        sizes: &[f64],
    ) -> &mut Self {
        let size_strs: Vec<String> = sizes.iter().map(|x| format_float(*x)).collect();
        self.lines.push(format!(
            "design \"{}\" {{ dim {}, basins {}, sizes [{}] }}",
            name,
            dim,
            n_basins,
            size_strs.join(", ")
        ));
        self
    }

    /// Emit raw .axol source line.
    pub fn raw(&mut self, line: &str) -> &mut Self {
        self.lines.push(line.to_string());
        self
    }

    /// Produce the final .axol source string.
    pub fn generate(&self) -> String {
        self.lines.join("\n") + "\n"
    }
}

fn format_float(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e9 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

// ---------------------------------------------------------------------------
// Predefined algorithm templates
// ---------------------------------------------------------------------------

/// Boolean encoding constants.
const TRUE_VEC: [f64; 2] = [0.1, 0.9];
const FALSE_VEC: [f64; 2] = [0.9, 0.1];

fn bool_vec(val: bool) -> [f64; 2] {
    if val { TRUE_VEC } else { FALSE_VEC }
}

/// Generate a NOT gate program.
pub fn gen_not(input: bool) -> String {
    let mut g = AxolGenerator::new();
    g.comment("AXOL NOT Gate");
    g.comment(&format!("Input: {} -> Expected: {}", input, !input));
    g.blank();
    g.gate("not", &[("x", &bool_vec(input))]);
    g.generate()
}

/// Generate an AND gate program.
pub fn gen_and(a: bool, b: bool) -> String {
    let mut g = AxolGenerator::new();
    g.comment("AXOL AND Gate");
    g.comment(&format!("Inputs: {} AND {} -> Expected: {}", a, b, a && b));
    g.blank();
    g.gate("and", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);
    g.generate()
}

/// Generate an OR gate program.
pub fn gen_or(a: bool, b: bool) -> String {
    let mut g = AxolGenerator::new();
    g.comment("AXOL OR Gate");
    g.comment(&format!("Inputs: {} OR {} -> Expected: {}", a, b, a || b));
    g.blank();
    g.gate("or", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);
    g.generate()
}

/// Generate XOR circuit: (A OR B) AND NOT(A AND B).
///
/// XOR requires composing multiple gates since it's not a primitive.
/// XOR(a,b) = AND(OR(a,b), NOT(AND(a,b)))
pub fn gen_xor(a: bool, b: bool) -> String {
    let mut g = AxolGenerator::new();
    g.comment("AXOL XOR Circuit");
    g.comment(&format!("XOR({}, {}) = {} ", a, b, a ^ b));
    g.comment("XOR(a,b) = AND(OR(a,b), NOT(AND(a,b)))");
    g.blank();

    g.comment("Step 1: A OR B");
    g.gate("or", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);
    g.blank();

    g.comment("Step 2: A AND B");
    g.gate("and", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);
    g.blank();

    // Result of AND(a,b)
    let and_result = a && b;
    g.comment("Step 3: NOT(A AND B)");
    g.gate("not", &[("x", &bool_vec(and_result))]);
    g.blank();

    // Final XOR = OR_result AND NAND_result
    let or_result = a || b;
    let nand_result = !and_result;
    g.comment("Step 4: AND(OR_result, NAND_result) = XOR");
    g.gate(
        "and",
        &[("a", &bool_vec(or_result)), ("b", &bool_vec(nand_result))],
    );
    g.generate()
}

/// Generate a half adder: Sum = XOR(a,b), Carry = AND(a,b).
pub fn gen_half_adder(a: bool, b: bool) -> String {
    let mut g = AxolGenerator::new();
    g.comment("AXOL Half Adder");
    g.comment(&format!("Inputs: A={}, B={}", a, b));
    g.comment(&format!(
        "Expected: Sum={}, Carry={}",
        a ^ b,
        a && b
    ));
    g.blank();

    // Carry = AND(a,b)
    g.comment("Carry = AND(A, B)");
    g.gate("and", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);
    g.blank();

    // Sum = XOR(a,b) = AND(OR(a,b), NOT(AND(a,b)))
    g.comment("Sum = XOR(A, B)");
    g.comment("  Step 1: OR(A, B)");
    g.gate("or", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);

    g.comment("  Step 2: AND(A, B)");
    g.gate("and", &[("a", &bool_vec(a)), ("b", &bool_vec(b))]);

    let and_ab = a && b;
    g.comment("  Step 3: NOT(AND(A, B))");
    g.gate("not", &[("x", &bool_vec(and_ab))]);

    let or_ab = a || b;
    let nand_ab = !and_ab;
    g.comment("  Step 4: AND(OR, NAND) = XOR = Sum");
    g.gate(
        "and",
        &[("a", &bool_vec(or_ab)), ("b", &bool_vec(nand_ab))],
    );
    g.generate()
}

/// Generate a binary classifier with confidence observation.
pub fn gen_binary_classifier(name: &str, dim: usize, omega: f64, phi: f64, seed: u64) -> String {
    let mut g = AxolGenerator::new();
    g.comment(&format!("AXOL Binary Classifier: {}", name));
    g.comment(&format!("dim={} omega={} phi={}", dim, omega, phi));
    g.blank();

    // Declare
    g.declare(
        name,
        &[("x", dim)],
        &["result"],
        &[("result", &["x"], "<~>")],
        Some((omega, phi)),
    );
    g.blank();

    // Weave
    g.weave(name, true, seed);
    g.blank();

    // Generate sample input
    let input: Vec<f64> = (0..dim)
        .map(|i| ((i as f64 * 0.7).sin() * 0.4 + 0.5).clamp(0.01, 0.99))
        .collect();

    g.comment("Observe with confidence voting");
    g.confident(name, 50, 0.95, &[("x", &input)]);
    g.generate()
}

/// Generate a multi-class classifier with basin design.
pub fn gen_multi_classifier(
    name: &str,
    dim: usize,
    n_classes: usize,
    omega: f64,
    phi: f64,
    seed: u64,
) -> String {
    let mut g = AxolGenerator::new();
    g.comment(&format!(
        "AXOL Multi-Class Classifier: {} ({} classes)",
        name, n_classes
    ));
    g.comment(&format!("dim={} omega={} phi={}", dim, omega, phi));
    g.blank();

    // Basin design for equal-sized classes
    let sizes: Vec<f64> = vec![1.0 / n_classes as f64; n_classes];
    g.design(name, dim, n_classes, &sizes);
    g.blank();

    // Declare
    g.declare(
        name,
        &[("x", dim)],
        &["class"],
        &[("class", &["x"], "<~>")],
        Some((omega, phi)),
    );
    g.blank();

    // Weave
    g.weave(name, true, seed);
    g.blank();

    // Sample input
    let input: Vec<f64> = (0..dim)
        .map(|i| ((i as f64 * 0.3).cos() * 0.4 + 0.5).clamp(0.01, 0.99))
        .collect();

    g.comment("Observe with confidence");
    g.confident(name, 100, 0.90, &[("x", &input)]);
    g.generate()
}

/// Generate a tapestry pipeline (chain composition).
pub fn gen_pipeline(
    name: &str,
    stage_names: &[&str],
    dim: usize,
    omega: f64,
    phi: f64,
    seed: u64,
) -> String {
    let mut g = AxolGenerator::new();
    g.comment(&format!("AXOL Pipeline: {}", name));
    g.comment(&format!(
        "Stages: {} -> composed into single matrix",
        stage_names.join(" -> ")
    ));
    g.blank();

    // Declare and weave each stage
    for (i, stage) in stage_names.iter().enumerate() {
        g.declare(
            stage,
            &[("x", dim)],
            &["y"],
            &[("y", &["x"], "<~>")],
            Some((omega, phi)),
        );
        g.weave(stage, true, seed + i as u64);
        g.blank();
    }

    // Compose
    g.compose(name, stage_names);
    g.blank();

    // Observe through composed pipeline
    let input: Vec<f64> = (0..dim)
        .map(|i| ((i as f64 * 0.5).sin() * 0.4 + 0.5).clamp(0.01, 0.99))
        .collect();

    g.comment("Observe through composed pipeline");
    // Note: currently observe works on individual tapestries
    // The compose command creates the chain in the runtime
    g.observe(stage_names[0], &[("x", &input)]);
    g.generate()
}

/// Generate convergent iteration program.
pub fn gen_convergent(
    name: &str,
    dim: usize,
    max_iter: usize,
    threshold: f64,
    omega: f64,
    phi: f64,
    seed: u64,
) -> String {
    let mut g = AxolGenerator::new();
    g.comment(&format!("AXOL Convergent Iteration: {}", name));
    g.comment(&format!(
        "Iterates until probability delta < {}",
        threshold
    ));
    g.blank();

    g.declare(
        name,
        &[("x", dim)],
        &["result"],
        &[("result", &["x"], "<~>")],
        Some((omega, phi)),
    );
    g.blank();

    g.weave(name, true, seed);
    g.blank();

    let input: Vec<f64> = (0..dim)
        .map(|i| ((i as f64 * 0.9).sin() * 0.4 + 0.5).clamp(0.01, 0.99))
        .collect();

    g.iterate(name, max_iter, "prob_delta", threshold, &[("x", &input)]);
    g.generate()
}

/// Generate a full algorithm program with all features.
pub fn gen_full_demo(seed: u64) -> String {
    let mut g = AxolGenerator::new();
    g.comment("=== AXOL Full Demo ===");
    g.comment("Demonstrates: declare, weave, observe, gates, confidence, iterate");
    g.blank();

    // 1. Logic gates
    g.comment("--- Logic Gates ---");
    g.gate("not", &[("x", &TRUE_VEC)]);
    g.gate("and", &[("a", &TRUE_VEC), ("b", &TRUE_VEC)]);
    g.gate("or", &[("a", &FALSE_VEC), ("b", &TRUE_VEC)]);
    g.blank();

    // 2. Declare + Weave + Observe
    g.comment("--- Declare/Weave/Observe ---");
    g.declare(
        "sensor",
        &[("x", 4)],
        &["class"],
        &[("class", &["x"], "<~>")],
        Some((0.9, 0.8)),
    );
    g.weave("sensor", true, seed);
    g.observe("sensor", &[("x", &[0.8, 0.2, 0.5, 0.3])]);
    g.blank();

    // 3. Confidence
    g.comment("--- Confident Observation ---");
    g.confident("sensor", 50, 0.95, &[("x", &[0.8, 0.2, 0.5, 0.3])]);
    g.blank();

    // 4. Iterate
    g.comment("--- Convergent Iteration ---");
    g.iterate(
        "sensor",
        30,
        "prob_delta",
        0.001,
        &[("x", &[0.8, 0.2, 0.5, 0.3])],
    );
    g.generate()
}

// ---------------------------------------------------------------------------
// Algorithm catalog (maps human-readable names to generators)
// ---------------------------------------------------------------------------

/// Available algorithm templates.
pub const ALGORITHM_CATALOG: &[(&str, &str)] = &[
    ("not", "NOT gate — boolean negation"),
    ("and", "AND gate — boolean conjunction"),
    ("or", "OR gate — boolean disjunction"),
    ("xor", "XOR circuit — exclusive or (composed from AND/OR/NOT)"),
    ("half-adder", "Half adder — Sum + Carry from two bits"),
    ("binary-classifier", "Binary classifier with confidence voting"),
    ("multi-classifier", "Multi-class classifier with basin design"),
    ("pipeline", "Tapestry pipeline — chained stages"),
    ("convergent", "Convergent iteration — iterates until stable"),
    ("demo", "Full demo — showcases all AXOL features"),
];

/// Generate .axol source for a named algorithm.
///
/// Returns `None` if the algorithm name is not recognized.
pub fn generate_algorithm(name: &str, seed: u64) -> Option<String> {
    match name {
        "not" => Some(gen_not(true)),
        "and" => Some(gen_and(true, true)),
        "or" => Some(gen_or(false, true)),
        "xor" => Some(gen_xor(true, false)),
        "half-adder" | "half_adder" | "halfadder" => Some(gen_half_adder(true, true)),
        "binary-classifier" | "binary_classifier" | "classifier" => {
            Some(gen_binary_classifier("classifier", 8, 0.9, 0.8, seed))
        }
        "multi-classifier" | "multi_classifier" | "multiclass" => {
            Some(gen_multi_classifier("classifier", 8, 3, 0.9, 0.7, seed))
        }
        "pipeline" | "chain" => Some(gen_pipeline(
            "pipeline",
            &["stage_a", "stage_b", "stage_c"],
            4,
            0.9,
            0.8,
            seed,
        )),
        "convergent" | "iterate" | "converge" => {
            Some(gen_convergent("convergent", 4, 50, 0.001, 0.9, 0.8, seed))
        }
        "demo" | "full" => Some(gen_full_demo(seed)),
        _ => None,
    }
}

/// List available algorithms as formatted string.
pub fn list_algorithms() -> String {
    let mut out = String::from("Available AXOL algorithms:\n\n");
    for (name, desc) in ALGORITHM_CATALOG {
        let _ = writeln!(out, "  {:<20} {}", name, desc);
    }
    out.push_str("\nUsage: axol generate <algorithm> [-o output.axol] [--seed N]\n");
    out
}
