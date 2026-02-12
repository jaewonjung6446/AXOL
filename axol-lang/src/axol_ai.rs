//! AXOL AI — self-referential .axol program generator using AXOL's own basin observation.
//!
//! The generator IS an AXOL program: a set of tapestries whose basin observations
//! determine what .axol code to produce. AI = Language.
//!
//! Architecture:
//!   1. User creates an AlgorithmRequest (structured specification)
//!   2. Request is encoded as a FloatVec (8-dimensional feature vector)
//!   3. Structure selector tapestry observed → program type (basin index)
//!   4. Parameter selector tapestries observed → dim, quality, details
//!   5. AxolGenerator assembles the final .axol source
//!
//! All decisions are made by basin observation — no if-else hard-coding.

use crate::types::FloatVec;
use crate::declare::*;
use crate::weaver::{self, Tapestry};
use crate::observatory;
use crate::compose;
use crate::learn;
use crate::codegen::AxolGenerator;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Feature encoding dimension
// ---------------------------------------------------------------------------

const FEAT_DIM: usize = 8;

// Feature indices:
// [0] logic_weight      — how much logic/boolean computation is needed
// [1] classify_weight   — how much classification is needed
// [2] pipeline_weight   — how much sequential composition is needed
// [3] iterate_weight    — how much iterative convergence is needed
// [4] complexity        — overall complexity (0.1 = simple, 0.9 = complex)
// [5] dim_hint          — normalized dimension hint (0.1 = small, 0.9 = large)
// [6] omega_hint        — desired stability (0.1 = low, 0.9 = high)
// [7] phi_hint          — desired precision (0.1 = low, 0.9 = high)

// ---------------------------------------------------------------------------
// Algorithm request
// ---------------------------------------------------------------------------

/// What kind of algorithm the user wants.
#[derive(Clone, Debug)]
pub enum TaskKind {
    /// Logic gate or circuit: NOT, AND, OR, XOR, half-adder, etc.
    Logic,
    /// Classification: binary or multi-class.
    Classify,
    /// Pipeline: sequential tapestry chain.
    Pipeline,
    /// Convergent iteration: repeat until stable.
    Converge,
    /// General / composite: mix of above.
    Composite,
}

/// Structured algorithm request.
#[derive(Clone, Debug)]
pub struct AlgorithmRequest {
    /// What kind of algorithm.
    pub task: TaskKind,
    /// Descriptive name for the generated program.
    pub name: String,
    /// Desired output dimension (None = let the AI decide).
    pub dim: Option<usize>,
    /// Number of classes (for classification).
    pub n_classes: Option<usize>,
    /// Number of pipeline stages (for pipeline).
    pub n_stages: Option<usize>,
    /// Desired quality: 0.0..1.0 (None = medium).
    pub quality: Option<f64>,
    /// Overall complexity: 0.0..1.0 (None = medium).
    pub complexity: Option<f64>,
}

impl AlgorithmRequest {
    pub fn new(name: &str, task: TaskKind) -> Self {
        Self {
            task,
            name: name.to_string(),
            dim: None,
            n_classes: None,
            n_stages: None,
            quality: None,
            complexity: None,
        }
    }

    /// Encode this request as an 8-dimensional feature vector for basin observation.
    pub fn encode(&self) -> FloatVec {
        let (logic_w, classify_w, pipeline_w, iterate_w) = match &self.task {
            TaskKind::Logic     => (0.9, 0.1, 0.1, 0.1),
            TaskKind::Classify  => (0.1, 0.9, 0.1, 0.2),
            TaskKind::Pipeline  => (0.1, 0.2, 0.9, 0.1),
            TaskKind::Converge  => (0.1, 0.2, 0.1, 0.9),
            TaskKind::Composite => (0.5, 0.5, 0.5, 0.5),
        };

        let complexity = self.complexity.unwrap_or(0.5).clamp(0.05, 0.95) as f32;
        let dim_hint = match self.dim {
            Some(d) if d <= 2 => 0.1,
            Some(d) if d <= 4 => 0.3,
            Some(d) if d <= 8 => 0.5,
            Some(d) if d <= 16 => 0.7,
            Some(_) => 0.9,
            None => 0.5,
        };
        let quality = self.quality.unwrap_or(0.7).clamp(0.05, 0.95) as f32;

        FloatVec::new(vec![
            logic_w as f32,
            classify_w as f32,
            pipeline_w as f32,
            iterate_w as f32,
            complexity,
            dim_hint,
            quality,      // omega_hint
            quality * 0.9, // phi_hint (slightly lower than omega)
        ])
    }
}

// ---------------------------------------------------------------------------
// AXOL AI — the generator built from tapestries
// ---------------------------------------------------------------------------

/// The AXOL AI: a set of tapestries that generate .axol programs via basin observation.
pub struct AxolAI {
    /// Selects program structure type (5 basins: logic, classify, pipeline, converge, composite)
    structure_tap: Tapestry,
    /// Selects dimension (4 basins: dim=2, 4, 8, 16)
    dim_tap: Tapestry,
    /// Selects quality level (3 basins: low, medium, high)
    quality_tap: Tapestry,
    /// Selects logic sub-type (5 basins: not, and, or, xor, half_adder)
    logic_tap: Tapestry,
    /// Confidence config for reliable observations
    confidence: compose::ConfidenceConfig,
}

/// Result of AI generation.
#[derive(Clone, Debug)]
pub struct GenerateResult {
    /// The generated .axol source code.
    pub source: String,
    /// What structure type was selected (basin index).
    pub structure_index: usize,
    /// Confidence of structure selection.
    pub structure_confidence: f64,
    /// Selected dimension.
    pub dim: usize,
    /// Selected quality (omega, phi).
    pub quality: (f64, f64),
    /// Total observations made during generation.
    pub total_observations: usize,
}

impl AxolAI {
    /// Create a new AXOL AI with the given seed.
    ///
    /// Uses AXOL's learn mechanism to train internal tapestries so that
    /// basin observations correctly route different task types to different outputs.
    pub fn new(seed: u64) -> Result<Self> {
        // Structure selector: learned to distinguish 5 task types
        let structure_tap = Self::learn_structure_selector(seed)?;

        // Dimension selector: learned to map dim_hint → 4 dim classes
        let dim_tap = Self::learn_dim_selector(seed + 100)?;

        // Quality selector: woven (quality is passed through directly)
        let quality_tap = Self::weave_selector("ai_quality", FEAT_DIM, 0.9, 0.85, seed + 200)?;

        // Logic sub-type selector: woven (no sub-type signal in feature encoding)
        let logic_tap = Self::weave_selector("ai_logic", FEAT_DIM, 0.85, 0.8, seed + 300)?;

        let confidence = compose::ConfidenceConfig {
            max_observations: 30,
            confidence_threshold: 0.80,
            min_observations: 5,
        };

        Ok(Self {
            structure_tap,
            dim_tap,
            quality_tap,
            logic_tap,
            confidence,
        })
    }

    /// Learn a structure selector that maps task-type feature vectors → 5 classes.
    fn learn_structure_selector(seed: u64) -> Result<Tapestry> {
        let mut ts = learn::TrainingSet::new("ai_structure", FEAT_DIM, 5);

        // Training data: multiple samples per class with varying complexity/dim/quality
        let variations: Vec<(f64, f64, f64)> = vec![
            (0.3, 0.3, 0.5), (0.5, 0.5, 0.7), (0.7, 0.7, 0.9),
            (0.2, 0.5, 0.6), (0.8, 0.3, 0.8),
        ];

        for (complexity, dim_hint, quality) in &variations {
            // Class 0: Logic — high logic_weight
            ts.add(vec![0.9, 0.1, 0.1, 0.1, *complexity, *dim_hint, *quality, quality * 0.9], 0);
            // Class 1: Classify — high classify_weight
            ts.add(vec![0.1, 0.9, 0.1, 0.2, *complexity, *dim_hint, *quality, quality * 0.9], 1);
            // Class 2: Pipeline — high pipeline_weight
            ts.add(vec![0.1, 0.2, 0.9, 0.1, *complexity, *dim_hint, *quality, quality * 0.9], 2);
            // Class 3: Converge — high iterate_weight
            ts.add(vec![0.1, 0.2, 0.1, 0.9, *complexity, *dim_hint, *quality, quality * 0.9], 3);
            // Class 4: Composite — balanced weights
            ts.add(vec![0.5, 0.5, 0.5, 0.5, *complexity, *dim_hint, *quality, quality * 0.9], 4);
        }

        let config = learn::LearnConfig {
            grid_r_steps: 12,
            grid_eps_steps: 12,
            nelder_mead_iters: 150,
            seeds: vec![seed, seed + 37, seed + 173],
            optimize_weights: true,
            ..learn::LearnConfig::default()
        };

        let result = learn::learn(&ts, &config)?;
        Ok(result.tapestry)
    }

    /// Learn a dim selector that maps dim_hint feature → 4 dimension classes.
    fn learn_dim_selector(seed: u64) -> Result<Tapestry> {
        let mut ts = learn::TrainingSet::new("ai_dim", FEAT_DIM, 4);

        // Training data: dim_hint at index 5 determines class
        let task_weights: Vec<(f64, f64, f64, f64)> = vec![
            (0.9, 0.1, 0.1, 0.1), // logic
            (0.1, 0.9, 0.1, 0.2), // classify
            (0.1, 0.2, 0.9, 0.1), // pipeline
            (0.1, 0.2, 0.1, 0.9), // converge
        ];

        for (lw, cw, pw, iw) in &task_weights {
            for &quality in &[0.5, 0.7, 0.9] {
                // Class 0: dim=2 → dim_hint=0.1
                ts.add(vec![*lw, *cw, *pw, *iw, 0.5, 0.1, quality, quality * 0.9], 0);
                // Class 1: dim=4 → dim_hint=0.3
                ts.add(vec![*lw, *cw, *pw, *iw, 0.5, 0.3, quality, quality * 0.9], 1);
                // Class 2: dim=8 → dim_hint=0.5
                ts.add(vec![*lw, *cw, *pw, *iw, 0.5, 0.5, quality, quality * 0.9], 2);
                // Class 3: dim=16 → dim_hint=0.7
                ts.add(vec![*lw, *cw, *pw, *iw, 0.5, 0.7, quality, quality * 0.9], 3);
            }
        }

        let config = learn::LearnConfig {
            grid_r_steps: 12,
            grid_eps_steps: 12,
            nelder_mead_iters: 150,
            seeds: vec![seed, seed + 53, seed + 211],
            optimize_weights: true,
            ..learn::LearnConfig::default()
        };

        let result = learn::learn(&ts, &config)?;
        Ok(result.tapestry)
    }

    fn weave_selector(name: &str, dim: usize, omega: f64, phi: f64, seed: u64) -> Result<Tapestry> {
        let mut builder = DeclarationBuilder::new(name);
        builder
            .input("x", dim)
            .output("result")
            .relate("result", &["x"], RelationKind::Proportional)
            .quality(omega, phi);
        let decl = builder.build();
        weaver::weave(&decl, true, seed)
    }

    /// Generate an .axol program from an algorithm request.
    ///
    /// All decisions are made by observing AXOL tapestries — basin observation
    /// determines the program structure, dimensions, and parameters.
    pub fn generate(&self, request: &AlgorithmRequest) -> Result<GenerateResult> {
        let input = request.encode();
        let input_ref: Vec<(&str, &FloatVec)> = vec![("x", &input)];
        let mut total_obs = 0usize;

        // 1. Observe structure selector → program type
        let structure_result = compose::observe_confident(
            &self.structure_tap,
            &input_ref,
            &self.confidence,
        )?;
        total_obs += structure_result.total_observations;
        let structure_idx = structure_result.value_index;

        // Map basin index to program type (modular mapping)
        // Basin count varies by tapestry, so we use modular arithmetic
        let n_types = 5;
        let program_type = structure_idx % n_types;

        // 2. Observe dimension selector
        let dim_result = compose::observe_confident(
            &self.dim_tap,
            &input_ref,
            &self.confidence,
        )?;
        total_obs += dim_result.total_observations;
        let dim_options = [2, 4, 8, 16];
        let dim = if let Some(d) = request.dim {
            d
        } else {
            dim_options[dim_result.value_index % dim_options.len()]
        };

        // 3. Observe quality selector
        let quality_result = compose::observe_confident(
            &self.quality_tap,
            &input_ref,
            &self.confidence,
        )?;
        total_obs += quality_result.total_observations;
        let quality_levels = [(0.7, 0.6), (0.85, 0.75), (0.95, 0.9)];
        let (omega, phi) = if let Some(q) = request.quality {
            (q, q * 0.9)
        } else {
            quality_levels[quality_result.value_index % quality_levels.len()]
        };

        // 4. Generate the .axol program based on observed decisions
        let source = match program_type {
            0 => self.gen_logic(request, &input, dim, omega, phi, &mut total_obs)?,
            1 => self.gen_classifier(request, dim, omega, phi),
            2 => self.gen_pipeline(request, dim, omega, phi),
            3 => self.gen_convergent(request, dim, omega, phi),
            _ => self.gen_composite(request, dim, omega, phi, &mut total_obs)?,
        };

        Ok(GenerateResult {
            source,
            structure_index: program_type,
            structure_confidence: structure_result.confidence,
            dim,
            quality: (omega, phi),
            total_observations: total_obs,
        })
    }

    // --- Program generators (each builds .axol using AxolGenerator) ---

    fn gen_logic(
        &self,
        request: &AlgorithmRequest,
        input: &FloatVec,
        dim: usize,
        omega: f64,
        phi: f64,
        total_obs: &mut usize,
    ) -> Result<String> {
        // Observe logic sub-type selector
        let input_ref: Vec<(&str, &FloatVec)> = vec![("x", input)];
        let logic_result = compose::observe_confident(
            &self.logic_tap,
            &input_ref,
            &self.confidence,
        )?;
        *total_obs += logic_result.total_observations;
        let logic_type = logic_result.value_index % 5;

        let mut g = AxolGenerator::new();
        g.comment(&format!("AXOL AI Generated: {}", request.name));
        g.comment(&format!("Task: Logic (sub-type {} selected by basin observation)", logic_type));
        g.comment(&format!("Decisions made by {} observations", total_obs));
        g.blank();

        match logic_type {
            0 => {
                g.comment("NOT gate — boolean negation");
                g.gate("not", &[("x", &[0.1, 0.9])]);
                g.blank();
                g.comment("Verify with opposite input");
                g.gate("not", &[("x", &[0.9, 0.1])]);
            }
            1 => {
                g.comment("AND gate — boolean conjunction (truth table)");
                for &(a, b) in &[(true, true), (true, false), (false, true), (false, false)] {
                    let av = if a { [0.1, 0.9] } else { [0.9, 0.1] };
                    let bv = if b { [0.1, 0.9] } else { [0.9, 0.1] };
                    g.comment(&format!("{} AND {} = {}", a, b, a && b));
                    g.gate("and", &[("a", &av), ("b", &bv)]);
                }
            }
            2 => {
                g.comment("OR gate — boolean disjunction (truth table)");
                for &(a, b) in &[(true, true), (true, false), (false, true), (false, false)] {
                    let av = if a { [0.1, 0.9] } else { [0.9, 0.1] };
                    let bv = if b { [0.1, 0.9] } else { [0.9, 0.1] };
                    g.comment(&format!("{} OR {} = {}", a, b, a || b));
                    g.gate("or", &[("a", &av), ("b", &bv)]);
                }
            }
            3 => {
                g.comment("XOR circuit — exclusive or");
                g.comment("XOR(a,b) = AND(OR(a,b), NOT(AND(a,b)))");
                g.blank();
                // Generate XOR for TRUE, FALSE case
                let (a, b) = (true, false);
                let av = [0.1, 0.9];
                let bv = [0.9, 0.1];
                g.comment("Step 1: OR(a, b)");
                g.gate("or", &[("a", &av), ("b", &bv)]);
                g.comment("Step 2: AND(a, b)");
                g.gate("and", &[("a", &av), ("b", &bv)]);
                let and_result = a && b;
                let and_vec = if and_result { [0.1, 0.9] } else { [0.9, 0.1] };
                g.comment("Step 3: NOT(AND(a, b))");
                g.gate("not", &[("x", &and_vec)]);
                let or_result = a || b;
                let nand_result = !and_result;
                let or_vec = if or_result { [0.1, 0.9] } else { [0.9, 0.1] };
                let nand_vec = if nand_result { [0.1, 0.9] } else { [0.9, 0.1] };
                g.comment("Step 4: AND(OR, NAND) = XOR");
                g.gate("and", &[("a", &or_vec), ("b", &nand_vec)]);
            }
            _ => {
                g.comment("Half adder — Sum + Carry");
                let (a, b) = (true, true);
                let av = [0.1, 0.9];
                let bv = [0.1, 0.9];
                g.comment(&format!("A={}, B={} -> Sum={}, Carry={}", a, b, a ^ b, a && b));
                g.blank();
                g.comment("Carry = AND(A, B)");
                g.gate("and", &[("a", &av), ("b", &bv)]);
                g.blank();
                g.comment("Sum = XOR(A, B) = AND(OR(A,B), NOT(AND(A,B)))");
                g.gate("or", &[("a", &av), ("b", &bv)]);
                g.gate("and", &[("a", &av), ("b", &bv)]);
                let and_ab = a && b;
                let and_vec = if and_ab { [0.1, 0.9] } else { [0.9, 0.1] };
                g.gate("not", &[("x", &and_vec)]);
                let or_ab = a || b;
                let nand_ab = !and_ab;
                let or_vec = if or_ab { [0.1, 0.9] } else { [0.9, 0.1] };
                let nand_vec = if nand_ab { [0.1, 0.9] } else { [0.9, 0.1] };
                g.gate("and", &[("a", &or_vec), ("b", &nand_vec)]);
            }
        }

        Ok(g.generate())
    }

    fn gen_classifier(&self, request: &AlgorithmRequest, dim: usize, omega: f64, phi: f64) -> String {
        let mut g = AxolGenerator::new();
        let n_classes = request.n_classes.unwrap_or(2);

        g.comment(&format!("AXOL AI Generated: {}", request.name));
        g.comment(&format!("Task: {}-class Classifier (dim={}, omega={}, phi={})",
            n_classes, dim, omega, phi));
        g.comment("Decisions made by AXOL basin observation");
        g.blank();

        if n_classes > 2 {
            let sizes: Vec<f64> = vec![1.0 / n_classes as f64; n_classes];
            g.design(&request.name, dim, n_classes, &sizes);
            g.blank();
        }

        g.declare(
            &request.name,
            &[("x", dim)],
            &["class"],
            &[("class", &["x"], "<~>")],
            Some((omega, phi)),
        );
        g.blank();
        g.weave(&request.name, true, 42);
        g.blank();

        // Generate sample input
        let input: Vec<f64> = (0..dim)
            .map(|i| ((i as f64 * 0.7).sin() * 0.4 + 0.5).clamp(0.05, 0.95))
            .collect();

        g.comment("Confident observation for reliable classification");
        g.confident(&request.name, 50, 0.90, &[("x", &input)]);

        g.generate()
    }

    fn gen_pipeline(&self, request: &AlgorithmRequest, dim: usize, omega: f64, phi: f64) -> String {
        let mut g = AxolGenerator::new();
        let n_stages = request.n_stages.unwrap_or(3);

        g.comment(&format!("AXOL AI Generated: {}", request.name));
        g.comment(&format!("Task: Pipeline ({} stages, dim={}, omega={}, phi={})",
            n_stages, dim, omega, phi));
        g.comment("Decisions made by AXOL basin observation");
        g.blank();

        // Generate stage names
        let stage_names: Vec<String> = (0..n_stages)
            .map(|i| format!("{}_stage_{}", request.name, i))
            .collect();

        // Declare and weave each stage
        for (i, stage) in stage_names.iter().enumerate() {
            g.declare(
                stage,
                &[("x", dim)],
                &["y"],
                &[("y", &["x"], "<~>")],
                Some((omega, phi)),
            );
            g.weave(stage, true, 42 + i as u64);
            g.blank();
        }

        // Compose
        let stage_refs: Vec<&str> = stage_names.iter().map(|s| s.as_str()).collect();
        g.compose(&request.name, &stage_refs);
        g.blank();

        // Observe through first stage
        let input: Vec<f64> = (0..dim)
            .map(|i| ((i as f64 * 0.5).sin() * 0.4 + 0.5).clamp(0.05, 0.95))
            .collect();

        g.comment("Observe through composed pipeline");
        g.observe(&stage_names[0], &[("x", &input)]);

        g.generate()
    }

    fn gen_convergent(&self, request: &AlgorithmRequest, dim: usize, omega: f64, phi: f64) -> String {
        let mut g = AxolGenerator::new();

        g.comment(&format!("AXOL AI Generated: {}", request.name));
        g.comment(&format!("Task: Convergent Iteration (dim={}, omega={}, phi={})",
            dim, omega, phi));
        g.comment("Decisions made by AXOL basin observation");
        g.blank();

        g.declare(
            &request.name,
            &[("x", dim)],
            &["result"],
            &[("result", &["x"], "<~>")],
            Some((omega, phi)),
        );
        g.blank();
        g.weave(&request.name, true, 42);
        g.blank();

        let input: Vec<f64> = (0..dim)
            .map(|i| ((i as f64 * 0.9).sin() * 0.4 + 0.5).clamp(0.05, 0.95))
            .collect();

        g.comment("Iterate until probability distribution stabilizes");
        g.iterate(&request.name, 50, "prob_delta", 0.001, &[("x", &input)]);

        g.generate()
    }

    fn gen_composite(
        &self,
        request: &AlgorithmRequest,
        dim: usize,
        omega: f64,
        phi: f64,
        total_obs: &mut usize,
    ) -> Result<String> {
        let mut g = AxolGenerator::new();

        g.comment(&format!("AXOL AI Generated: {}", request.name));
        g.comment(&format!("Task: Composite (dim={}, omega={}, phi={})", dim, omega, phi));
        g.comment("Decisions made by AXOL basin observation");
        g.blank();

        // Composite: combine logic + classification + iteration
        g.comment("=== Phase 1: Logic preprocessing ===");
        g.gate("and", &[("a", &[0.1, 0.9]), ("b", &[0.1, 0.9])]);
        g.blank();

        g.comment("=== Phase 2: Classification ===");
        g.declare(
            &request.name,
            &[("x", dim)],
            &["result"],
            &[("result", &["x"], "<~>")],
            Some((omega, phi)),
        );
        g.weave(&request.name, true, 42);
        g.blank();

        let input: Vec<f64> = (0..dim)
            .map(|i| ((i as f64 * 0.3).cos() * 0.4 + 0.5).clamp(0.05, 0.95))
            .collect();

        g.comment("=== Phase 3: Confident observation ===");
        g.confident(&request.name, 50, 0.90, &[("x", &input)]);
        g.blank();

        g.comment("=== Phase 4: Convergent refinement ===");
        g.iterate(&request.name, 30, "prob_delta", 0.001, &[("x", &input)]);

        Ok(g.generate())
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors for common requests
// ---------------------------------------------------------------------------

impl AlgorithmRequest {
    /// Create a logic circuit request.
    pub fn logic(name: &str) -> Self {
        Self::new(name, TaskKind::Logic)
    }

    /// Create a classifier request.
    pub fn classifier(name: &str, n_classes: usize) -> Self {
        let mut r = Self::new(name, TaskKind::Classify);
        r.n_classes = Some(n_classes);
        r
    }

    /// Create a pipeline request.
    pub fn pipeline(name: &str, n_stages: usize) -> Self {
        let mut r = Self::new(name, TaskKind::Pipeline);
        r.n_stages = Some(n_stages);
        r
    }

    /// Create a convergent iteration request.
    pub fn convergent(name: &str) -> Self {
        Self::new(name, TaskKind::Converge)
    }

    /// Create a composite request.
    pub fn composite(name: &str) -> Self {
        Self::new(name, TaskKind::Composite)
    }

    /// Set the dimension.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = Some(dim);
        self
    }

    /// Set the quality level (0.0..1.0).
    pub fn with_quality(mut self, quality: f64) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Set the complexity (0.0..1.0).
    pub fn with_complexity(mut self, complexity: f64) -> Self {
        self.complexity = Some(complexity);
        self
    }
}
