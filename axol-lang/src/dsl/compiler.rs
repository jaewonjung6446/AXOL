//! AXOL DSL Compiler — executes parsed AST through the Declare/Weave/Observe pipeline.

use std::collections::HashMap;
use std::time::Instant;

use crate::types::{FloatVec, BasinStructure};
use crate::declare::*;
use crate::wave::{Wave, InterferencePattern};
use crate::weaver::{self, Tapestry};
use crate::observatory::{self, Observation};
use crate::compose;
use crate::learn;
use crate::relation::{self, Relation, Expectation};
use crate::dsl::parser::*;
use crate::errors::{AxolError, Result};

pub struct Runtime {
    declarations: HashMap<String, EntangleDeclaration>,
    tapestries: HashMap<String, Tapestry>,
    chains: HashMap<String, compose::TapestryChain>,
    basin_designs: HashMap<String, compose::BasinDesign>,
    basin_structures: HashMap<String, BasinStructure>,
    waves: HashMap<String, Wave>,
    relations: HashMap<String, Relation>,
    observations: HashMap<String, (Wave, Observation)>,
    expectations: HashMap<String, Expectation>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            tapestries: HashMap::new(),
            chains: HashMap::new(),
            basin_designs: HashMap::new(),
            basin_structures: HashMap::new(),
            waves: HashMap::new(),
            relations: HashMap::new(),
            observations: HashMap::new(),
            expectations: HashMap::new(),
        }
    }

    pub fn execute(&mut self, program: &Program) -> Result<Vec<String>> {
        let mut output_lines = Vec::new();
        let total_start = Instant::now();

        for stmt in &program.statements {
            match stmt {
                Statement::Declare(decl) => {
                    self.exec_declare(decl, &mut output_lines)?;
                }
                Statement::Weave(cmd) => {
                    self.exec_weave(cmd, &mut output_lines)?;
                }
                Statement::Observe(cmd) => {
                    self.exec_observe(cmd, &mut output_lines)?;
                }
                Statement::Reobserve(cmd) => {
                    self.exec_reobserve(cmd, &mut output_lines)?;
                }
                Statement::ComposeChain(cmd) => {
                    self.exec_compose(cmd, &mut output_lines)?;
                }
                Statement::GateOp(cmd) => {
                    self.exec_gate(cmd, &mut output_lines)?;
                }
                Statement::ConfidentObs(cmd) => {
                    self.exec_confident(cmd, &mut output_lines)?;
                }
                Statement::IterateObs(cmd) => {
                    self.exec_iterate(cmd, &mut output_lines)?;
                }
                Statement::DesignBasins(cmd) => {
                    self.exec_design(cmd, &mut output_lines)?;
                }
                Statement::Learn(cmd) => {
                    self.exec_learn(cmd, &mut output_lines)?;
                }
                Statement::DefineBasinsCmd(cmd) => {
                    self.exec_define_basins(cmd, &mut output_lines)?;
                }
                Statement::WaveCreate(cmd) => {
                    self.exec_wave(cmd, &mut output_lines)?;
                }
                Statement::FocusWave(cmd) => {
                    self.exec_focus(cmd, &mut output_lines)?;
                }
                Statement::GazeWave(cmd) => {
                    self.exec_gaze(cmd, &mut output_lines)?;
                }
                Statement::GlimpseWave(cmd) => {
                    self.exec_glimpse(cmd, &mut output_lines)?;
                }
                Statement::RelDeclare(cmd) => {
                    self.exec_rel(cmd, &mut output_lines)?;
                }
                Statement::ExpectDeclare(cmd) => {
                    self.exec_expect(cmd, &mut output_lines)?;
                }
                Statement::WidenWave(cmd) => {
                    self.exec_widen(cmd, &mut output_lines)?;
                }
                Statement::ResolveObs(cmd) => {
                    self.exec_resolve(cmd, &mut output_lines)?;
                }
            }
        }

        let elapsed = total_start.elapsed();
        output_lines.push(format!("--- Total: {:.3}ms ---", elapsed.as_secs_f64() * 1000.0));
        Ok(output_lines)
    }

    fn exec_declare(&mut self, block: &DeclareBlock, out: &mut Vec<String>) -> Result<()> {
        let mut builder = DeclarationBuilder::new(&block.name);

        for inp in &block.inputs {
            builder.input(&inp.name, inp.dim);
        }
        for output in &block.outputs {
            builder.output(output);
        }
        for rel in &block.relations {
            let kind = RelationKind::from_str(&rel.relation)
                .ok_or_else(|| AxolError::Parse(format!("Unknown relation: {}", rel.relation)))?;
            let sources: Vec<&str> = rel.sources.iter().map(|s| s.as_str()).collect();
            builder.relate(&rel.target, &sources, kind);
        }
        if let Some(ref q) = block.quality {
            builder.quality(q.omega, q.phi);
        }

        let decl = builder.build();
        out.push(format!("[declare] '{}': {} inputs, {} relations, {} outputs",
            block.name, decl.inputs.len(), decl.relations.len(), decl.outputs.len()));
        self.declarations.insert(block.name.clone(), decl);
        Ok(())
    }

    fn exec_weave(&mut self, cmd: &WeaveCmd, out: &mut Vec<String>) -> Result<()> {
        let decl = self.declarations.get(&cmd.name)
            .ok_or_else(|| AxolError::Weaver(format!("Declaration '{}' not found", cmd.name)))?
            .clone();

        let start = Instant::now();
        let mut tapestry = weaver::weave(&decl, cmd.quantum, cmd.seed)?;

        // If from_basins specified, override the basin_structure
        if let Some(ref bs_name) = cmd.from_basins {
            if let Some(bs) = self.basin_structures.get(bs_name) {
                tapestry.basin_structure = bs.clone();
                // Update report with overridden basin structure
                tapestry.report.estimated_omega = bs.omega();
                tapestry.report.estimated_phi = bs.phi();
                tapestry.report.shannon_entropy = bs.shannon_entropy();
                tapestry.report.n_basins = bs.n_basins;
                tapestry.preserve_basins = true;
            } else {
                return Err(AxolError::Weaver(format!("BasinStructure '{}' not found", bs_name)));
            }
        }

        let elapsed = start.elapsed();

        out.push(format!(
            "[weave] '{}': quantum={} seed={} omega={:.4} phi={:.4} basins={} ({:.3}ms)",
            cmd.name, cmd.quantum, cmd.seed,
            tapestry.report.estimated_omega,
            tapestry.report.estimated_phi,
            tapestry.report.n_basins,
            elapsed.as_secs_f64() * 1000.0,
        ));

        if let Some(ref dm) = tapestry.density_matrix {
            out.push(format!("  density: purity={:.4} dim={}", dm.purity(), dm.dim));
        }

        self.tapestries.insert(cmd.name.clone(), tapestry);
        Ok(())
    }

    fn exec_observe(&mut self, cmd: &ObserveCmd, out: &mut Vec<String>) -> Result<()> {
        // Check if observing a relation
        if self.relations.contains_key(&cmd.name) {
            if let Some(ref expect_name) = cmd.with_expect {
                // Observe relation with expectation landscape
                let expect = self.expectations.get(expect_name)
                    .ok_or_else(|| AxolError::Relation(format!("Expectation '{}' not found", expect_name)))?
                    .clone();

                let rel = self.relations.get_mut(&cmd.name).unwrap();

                let start = Instant::now();
                let result = rel.apply_expect(&expect)?;
                let elapsed = start.elapsed();

                out.push(format!(
                    "[observe] rel '{}' with '{}': idx={} alignment={:.4} negativity_delta={:+.4} coherences={} ({:.1}us)",
                    cmd.name, expect_name, result.value_index, result.alignment,
                    result.negativity_delta, result.surviving_coherences,
                    elapsed.as_secs_f64() * 1e6,
                ));

                let top: Vec<String> = result.probabilities.iter().enumerate()
                    .take(5)
                    .map(|(i, p)| format!("[{}]={:.4}", i, p))
                    .collect();
                out.push(format!("  probs: {}", top.join(", ")));
                out.push(format!("  negativity: {:.4}", rel.negativity));

                if let Some(ref var) = cmd.result_var {
                    self.waves.insert(var.clone(), result.wave);
                }
                return Ok(());
            } else {
                // Observe relation without expectation — just read its state
                let rel = self.relations.get(&cmd.name).unwrap();
                let probs = rel.gaze();
                let (value_index, _) = rel.wave.observe();

                let top: Vec<String> = probs.iter().enumerate()
                    .take(5)
                    .map(|(i, p)| format!("[{}]={:.4}", i, p))
                    .collect();

                out.push(format!(
                    "[observe] rel '{}': idx={} negativity={:.4}",
                    cmd.name, value_index, rel.negativity,
                ));
                out.push(format!("  probs: {}", top.join(", ")));

                if let Some(ref var) = cmd.result_var {
                    self.waves.insert(var.clone(), rel.wave.clone());
                }
                return Ok(());
            }
        }

        let tapestry = self.tapestries.get(&cmd.name)
            .ok_or_else(|| AxolError::Observatory(format!("Tapestry '{}' not woven", cmd.name)))?;

        let input_vecs: Vec<(String, FloatVec)> = cmd.inputs.iter().map(|(name, vals)| {
            (name.clone(), FloatVec::new(vals.iter().map(|&v| v as f32).collect()))
        }).collect();
        let inputs: Vec<(&str, &FloatVec)> = input_vecs.iter().map(|(n, v)| (n.as_str(), v)).collect();

        let start = Instant::now();
        let obs = observatory::observe(tapestry, &inputs)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[observe] '{}': idx={} label={:?} omega={:.4} phi={:.4} ({:.1}us)",
            cmd.name, obs.value_index, obs.value_label,
            obs.omega, obs.phi,
            elapsed.as_secs_f64() * 1e6,
        ));

        self.print_observation_details(&obs, out);

        // Store observation if result_var is set
        if let Some(ref var) = cmd.result_var {
            if let Some(ref wave) = obs.wave {
                self.waves.insert(var.clone(), wave.clone());
            }
            self.observations.insert(var.clone(), (
                obs.wave.clone().unwrap_or_else(|| Wave::collapsed(obs.probabilities.dim(), obs.value_index)),
                obs,
            ));
        }

        Ok(())
    }

    fn exec_reobserve(&mut self, cmd: &ReobserveCmd, out: &mut Vec<String>) -> Result<()> {
        let tapestry = self.tapestries.get(&cmd.name)
            .ok_or_else(|| AxolError::Observatory(format!("Tapestry '{}' not woven", cmd.name)))?;

        let input_vecs: Vec<(String, FloatVec)> = cmd.inputs.iter().map(|(name, vals)| {
            (name.clone(), FloatVec::new(vals.iter().map(|&v| v as f32).collect()))
        }).collect();
        let inputs: Vec<(&str, &FloatVec)> = input_vecs.iter().map(|(n, v)| (n.as_str(), v)).collect();

        let start = Instant::now();
        let obs = observatory::reobserve(tapestry, &inputs, cmd.count)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[reobserve] '{}' x{}: idx={} omega={:.4} phi={:.4} ({:.3}ms)",
            cmd.name, cmd.count, obs.value_index,
            obs.omega, obs.phi,
            elapsed.as_secs_f64() * 1000.0,
        ));

        self.print_observation_details(&obs, out);
        Ok(())
    }

    // --- Compose layer ---

    fn exec_compose(&mut self, cmd: &ComposeCmd, out: &mut Vec<String>) -> Result<()> {
        let mut stages = Vec::new();
        for stage_name in &cmd.stages {
            let t = self.tapestries.get(stage_name)
                .ok_or_else(|| AxolError::Compose(format!("Tapestry '{}' not found", stage_name)))?;
            stages.push(t.clone());
        }

        let start = Instant::now();
        let chain = compose::chain(&cmd.name, stages)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[compose] '{}': {} stages, matrix {}x{} ({:.3}ms)",
            cmd.name, chain.stages.len(),
            chain.composed_matrix.rows, chain.composed_matrix.cols,
            elapsed.as_secs_f64() * 1000.0,
        ));

        self.chains.insert(cmd.name.clone(), chain);
        Ok(())
    }

    fn exec_gate(&mut self, cmd: &GateCmd, out: &mut Vec<String>) -> Result<()> {
        let start = Instant::now();

        let obs = match cmd.gate_type.as_str() {
            "not" => {
                let input = cmd.inputs.first()
                    .ok_or_else(|| AxolError::Compose("NOT gate needs 1 input".into()))?;
                let fv = FloatVec::new(input.1.iter().map(|&v| v as f32).collect());
                compose::gate_not(&fv)?
            }
            "and" => {
                if cmd.inputs.len() < 2 {
                    return Err(AxolError::Compose("AND gate needs 2 inputs".into()));
                }
                let a = FloatVec::new(cmd.inputs[0].1.iter().map(|&v| v as f32).collect());
                let b = FloatVec::new(cmd.inputs[1].1.iter().map(|&v| v as f32).collect());
                compose::gate_and(&a, &b)?
            }
            "or" => {
                if cmd.inputs.len() < 2 {
                    return Err(AxolError::Compose("OR gate needs 2 inputs".into()));
                }
                let a = FloatVec::new(cmd.inputs[0].1.iter().map(|&v| v as f32).collect());
                let b = FloatVec::new(cmd.inputs[1].1.iter().map(|&v| v as f32).collect());
                compose::gate_or(&a, &b)?
            }
            other => {
                return Err(AxolError::Compose(format!("Unknown gate type: {}", other)));
            }
        };

        let elapsed = start.elapsed();
        out.push(format!(
            "[gate] {}: idx={} ({:.1}us)",
            cmd.gate_type, obs.value_index,
            elapsed.as_secs_f64() * 1e6,
        ));
        self.print_observation_details(&obs, out);
        Ok(())
    }

    fn exec_confident(&mut self, cmd: &ConfidentCmd, out: &mut Vec<String>) -> Result<()> {
        let tapestry = self.tapestries.get(&cmd.name)
            .ok_or_else(|| AxolError::Compose(format!("Tapestry '{}' not found", cmd.name)))?;

        let input_vecs: Vec<(String, FloatVec)> = cmd.inputs.iter().map(|(name, vals)| {
            (name.clone(), FloatVec::new(vals.iter().map(|&v| v as f32).collect()))
        }).collect();
        let inputs: Vec<(&str, &FloatVec)> = input_vecs.iter()
            .map(|(n, v)| (n.as_str(), v)).collect();

        let config = compose::ConfidenceConfig {
            max_observations: cmd.max_observations,
            confidence_threshold: cmd.threshold,
            min_observations: 5,
        };

        let start = Instant::now();
        let result = compose::observe_confident(tapestry, &inputs, &config)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[confident] '{}': idx={} confidence={:.4} obs={} early_stopped={} ({:.3}ms)",
            cmd.name, result.value_index, result.confidence,
            result.total_observations, result.early_stopped,
            elapsed.as_secs_f64() * 1000.0,
        ));

        let top: Vec<String> = result.avg_probabilities.iter().enumerate()
            .take(5)
            .map(|(i, p)| format!("[{}]={:.4}", i, p))
            .collect();
        out.push(format!("  avg_probs: {}", top.join(", ")));

        Ok(())
    }

    fn exec_iterate(&mut self, cmd: &IterateCmd, out: &mut Vec<String>) -> Result<()> {
        let tapestry = self.tapestries.get_mut(&cmd.name)
            .ok_or_else(|| AxolError::Compose(format!("Tapestry '{}' not found", cmd.name)))?;

        let input_vecs: Vec<(String, FloatVec)> = cmd.inputs.iter().map(|(name, vals)| {
            (name.clone(), FloatVec::new(vals.iter().map(|&v| v as f32).collect()))
        }).collect();
        let inputs: Vec<(&str, &FloatVec)> = input_vecs.iter()
            .map(|(n, v)| (n.as_str(), v)).collect();

        let convergence = match cmd.converge_type.as_str() {
            "prob_delta" => compose::ConvergenceCriterion::ProbabilityDelta(cmd.converge_value),
            "stable_index" => compose::ConvergenceCriterion::StableIndex(cmd.converge_value as usize),
            "omega_target" => compose::ConvergenceCriterion::OmegaTarget,
            "phi_target" => compose::ConvergenceCriterion::PhiTarget,
            "purity" => compose::ConvergenceCriterion::PurityThreshold,
            "basin_dist" => compose::ConvergenceCriterion::BasinDistribution(cmd.converge_value),
            _ => compose::ConvergenceCriterion::ProbabilityDelta(cmd.converge_value),
        };

        let config = compose::IterateConfig {
            max_iterations: cmd.max_iterations,
            min_iterations: 3,
            convergence,
            feedback: true,
            feedback_strength: 0.5,
        };

        let start = Instant::now();
        let result = compose::iterate(tapestry, &inputs, &config)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[iterate] '{}': iters={} converged={} delta={:.6} ({:.3}ms)",
            cmd.name, result.iterations, result.converged, result.final_delta,
            elapsed.as_secs_f64() * 1000.0,
        ));

        self.print_observation_details(&result.observation, out);
        Ok(())
    }

    fn exec_design(&mut self, cmd: &DesignCmd, out: &mut Vec<String>) -> Result<()> {
        let spec = compose::BasinSpec {
            n_basins: cmd.n_basins,
            target_sizes: cmd.sizes.clone(),
            boundary_hints: Vec::new(),
            dim: cmd.dim,
        };

        let config = compose::BasinDesignConfig::default();

        let start = Instant::now();
        let design = compose::design_basins(&spec, &config)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[design] '{}': {} basins found, score={:.4}, {} evals ({:.1}ms)",
            cmd.name, design.basins.len(), design.score, design.iterations,
            elapsed.as_secs_f64() * 1000.0,
        ));

        for (i, b) in design.basins.iter().enumerate().take(5) {
            out.push(format!("  basin {}: size={:.4} lyap={:.4}", i, b.size, b.local_lyapunov));
        }

        self.basin_designs.insert(cmd.name.clone(), design);
        Ok(())
    }

    fn exec_learn(&mut self, cmd: &LearnCmd, out: &mut Vec<String>) -> Result<()> {
        let mut training = learn::TrainingSet::new(&cmd.name, cmd.dim, 0);
        for (values, expected) in &cmd.samples {
            training.add(values.clone(), *expected);
        }
        training.n_classes = cmd.samples.iter().map(|(_, e)| *e).max().unwrap_or(0) + 1;

        let config = learn::LearnConfig {
            quantum: cmd.quantum,
            seeds: vec![cmd.seed, cmd.seed + 81, cmd.seed + 235],
            ..learn::LearnConfig::default()
        };

        let start = Instant::now();
        let result = learn::learn(&training, &config)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[learn] '{}': accuracy={:.1}% r={:.4} eps={:.4} evals={} ({:.1}ms)",
            cmd.name,
            result.accuracy * 100.0,
            result.best_r,
            result.best_epsilon,
            result.total_evaluations,
            elapsed.as_secs_f64() * 1000.0,
        ));

        self.tapestries.insert(cmd.name.clone(), result.tapestry);
        Ok(())
    }

    fn exec_define_basins(&mut self, cmd: &DefineBasinsBlock, out: &mut Vec<String>) -> Result<()> {
        let centroids: Vec<Vec<f64>> = cmd.basins.iter().map(|b| b.centroid.clone()).collect();
        let volumes: Vec<f64> = cmd.basins.iter().map(|b| b.volume).collect();

        let bs = BasinStructure::from_direct(
            cmd.dim,
            centroids,
            volumes,
            cmd.fractal_dim,
        );

        out.push(format!(
            "[define_basins] '{}': dim={} basins={} omega={:.4} phi={:.4} entropy={:.4}",
            cmd.name, bs.dim, bs.n_basins, bs.omega(), bs.phi(), bs.shannon_entropy(),
        ));

        self.basin_structures.insert(cmd.name.clone(), bs);
        Ok(())
    }

    // --- Wave system ---

    fn exec_wave(&mut self, cmd: &WaveCmd, out: &mut Vec<String>) -> Result<()> {
        let tapestry = self.tapestries.get(&cmd.tapestry_name)
            .ok_or_else(|| AxolError::Observatory(format!("Tapestry '{}' not woven", cmd.tapestry_name)))?;

        let input_vecs: Vec<(String, FloatVec)> = cmd.inputs.iter().map(|(name, vals)| {
            (name.clone(), FloatVec::new(vals.iter().map(|&v| v as f32).collect()))
        }).collect();
        let inputs: Vec<(&str, &FloatVec)> = input_vecs.iter().map(|(n, v)| (n.as_str(), v)).collect();

        let start = Instant::now();
        let wave = observatory::gaze(tapestry, &inputs)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[wave] '{}' = gaze('{}') -> {} (C=0, {:.1}us)",
            cmd.var_name, cmd.tapestry_name, wave,
            elapsed.as_secs_f64() * 1e6,
        ));

        self.waves.insert(cmd.var_name.clone(), wave);
        Ok(())
    }

    fn exec_focus(&mut self, cmd: &FocusCmd, out: &mut Vec<String>) -> Result<()> {
        let wave = self.waves.get(&cmd.var_name)
            .ok_or_else(|| AxolError::Observatory(format!("Wave '{}' not found", cmd.var_name)))?
            .clone();

        let start = Instant::now();
        let focused = wave.focus(cmd.gamma);
        let elapsed = start.elapsed();

        out.push(format!(
            "[focus] '{}' gamma={:.2} -> t={:.3} dominant={} (C={:.2}, {:.1}us)",
            cmd.var_name, cmd.gamma, focused.t, focused.dominant(), cmd.gamma,
            elapsed.as_secs_f64() * 1e6,
        ));

        self.waves.insert(cmd.var_name.clone(), focused);
        Ok(())
    }

    fn exec_gaze(&mut self, cmd: &GazeCmd, out: &mut Vec<String>) -> Result<()> {
        let wave = self.waves.get(&cmd.var_name)
            .ok_or_else(|| AxolError::Observatory(format!("Wave '{}' not found", cmd.var_name)))?;

        let probs = wave.gaze();
        let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = indexed.iter().take(5)
            .map(|(i, p)| format!("[{}]={:.4}", i, p))
            .collect();

        out.push(format!(
            "[gaze] '{}': t={:.3} probs: {} (C=0)",
            cmd.var_name, wave.t, top.join(", "),
        ));

        Ok(())
    }

    fn exec_glimpse(&mut self, cmd: &GlimpseCmd, out: &mut Vec<String>) -> Result<()> {
        let wave = self.waves.get(&cmd.var_name)
            .ok_or_else(|| AxolError::Observatory(format!("Wave '{}' not found", cmd.var_name)))?
            .clone();

        let start = Instant::now();
        let glimpsed = wave.focus(cmd.gamma);
        let elapsed = start.elapsed();

        let probs = glimpsed.gaze();
        let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = indexed.iter().take(5)
            .map(|(i, p)| format!("[{}]={:.4}", i, p))
            .collect();

        out.push(format!(
            "[glimpse] '{}' gamma={:.2}: t={:.3} probs: {} (C={:.2}, {:.1}us)",
            cmd.var_name, cmd.gamma, glimpsed.t, top.join(", "), cmd.gamma,
            elapsed.as_secs_f64() * 1e6,
        ));

        self.waves.insert(cmd.var_name.clone(), glimpsed);
        Ok(())
    }

    // --- Relation-first (v2) ---

    fn exec_rel(&mut self, cmd: &RelDecl, out: &mut Vec<String>) -> Result<()> {
        // Look up from/to waves
        let from_wave = self.waves.get(&cmd.from)
            .ok_or_else(|| AxolError::Relation(format!("Wave '{}' not found", cmd.from)))?
            .clone();
        let to_wave = self.waves.get(&cmd.to)
            .ok_or_else(|| AxolError::Relation(format!("Wave '{}' not found", cmd.to)))?
            .clone();

        // Determine interference pattern from via clause
        let pattern = if let Some(ref via) = cmd.via {
            let kind = RelationKind::from_str(via)
                .ok_or_else(|| AxolError::Parse(format!("Unknown relation kind: {}", via)))?;
            InterferencePattern::from_relation(&kind)
        } else {
            InterferencePattern::Constructive // default
        };

        let dir_str = match cmd.direction {
            RelDirection::Bidir => "<->",
            RelDirection::Forward => "<-",
            RelDirection::Conflict => "><",
        };

        let start = Instant::now();
        let rel = Relation::new(
            &cmd.name, &cmd.from, &cmd.to,
            cmd.direction.clone(), &from_wave, &to_wave, pattern,
        )?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[rel] '{}' = {} {} {} negativity={:.4} ({:.1}us)",
            cmd.name, cmd.from, dir_str, cmd.to, rel.negativity,
            elapsed.as_secs_f64() * 1e6,
        ));

        let probs = rel.gaze();
        let top: Vec<String> = probs.iter().enumerate()
            .take(5)
            .map(|(i, p)| format!("[{}]={:.4}", i, p))
            .collect();
        out.push(format!("  probs: {}", top.join(", ")));

        self.relations.insert(cmd.name.clone(), rel);
        Ok(())
    }

    fn exec_expect(&mut self, cmd: &ExpectDecl, out: &mut Vec<String>) -> Result<()> {
        let landscape = match &cmd.landscape {
            ExpectLandscape::Distribution(values) => values.clone(),
            ExpectLandscape::WaveRef(wave_name) => {
                let wave = self.waves.get(wave_name)
                    .ok_or_else(|| AxolError::Relation(format!("Wave '{}' not found for expect landscape", wave_name)))?;
                wave.probabilities()
            }
        };

        let expect = Expectation::from_distribution(&cmd.name, landscape.clone(), cmd.strength);

        let source_desc = match &cmd.landscape {
            ExpectLandscape::Distribution(_) => format!("[{}]", landscape.iter().take(5).map(|v| format!("{:.2}", v)).collect::<Vec<_>>().join(", ")),
            ExpectLandscape::WaveRef(name) => format!("wave '{}'", name),
        };

        out.push(format!(
            "[expect] '{}' = {} strength={:.2} (landscape dim={})",
            cmd.name, source_desc, cmd.strength, landscape.len(),
        ));

        self.expectations.insert(cmd.name.clone(), expect);
        Ok(())
    }

    fn exec_widen(&mut self, cmd: &WidenCmd, out: &mut Vec<String>) -> Result<()> {
        // Try relation first, then wave
        if let Some(rel) = self.relations.get_mut(&cmd.var_name) {
            let old_neg = rel.negativity;
            let old_t = rel.wave.t;

            let start = Instant::now();
            rel.widen(cmd.amount)?;
            let elapsed = start.elapsed();

            out.push(format!(
                "[widen] rel '{}' amount={:.2}: t={:.3}->{:.3} negativity={:.4}->{:.4} ({:.1}us)",
                cmd.var_name, cmd.amount, old_t, rel.wave.t,
                old_neg, rel.negativity, elapsed.as_secs_f64() * 1e6,
            ));
            return Ok(());
        }

        if let Some(wave) = self.waves.get(&cmd.var_name) {
            let old_t = wave.t;
            let amount = cmd.amount.clamp(0.0, 1.0);

            let start = Instant::now();
            // Widen: reduce t and apply depolarizing channel
            let new_t = old_t * (1.0 - amount);
            let rho = wave.to_density();
            let kraus = crate::density::depolarizing_channel(wave.dim, amount);
            let widened_rho = crate::density::apply_channel(&rho, &kraus);
            let mut new_wave = Wave::from_density(widened_rho);
            new_wave.t = new_t;
            let elapsed = start.elapsed();

            out.push(format!(
                "[widen] wave '{}' amount={:.2}: t={:.3}->{:.3} ({:.1}us)",
                cmd.var_name, cmd.amount, old_t, new_wave.t,
                elapsed.as_secs_f64() * 1e6,
            ));

            self.waves.insert(cmd.var_name.clone(), new_wave);
            return Ok(());
        }

        Err(AxolError::Relation(format!(
            "No relation or wave named '{}' found", cmd.var_name
        )))
    }

    fn exec_resolve(&mut self, cmd: &ResolveCmd, out: &mut Vec<String>) -> Result<()> {
        if cmd.observations.len() < 2 {
            return Err(AxolError::Relation("Resolve needs at least 2 observations".into()));
        }

        // Look up waves from observations or waves map
        let wave_a = self.get_resolve_wave(&cmd.observations[0])?;
        let wave_b = self.get_resolve_wave(&cmd.observations[1])?;

        let strategy_name = match cmd.strategy {
            ResolveStrategy::Interfere => "interfere",
            ResolveStrategy::Branch => "branch",
            ResolveStrategy::Rebase(_) => "rebase",
            ResolveStrategy::Superpose => "superpose",
        };

        let start = Instant::now();
        let resolved = match &cmd.strategy {
            ResolveStrategy::Interfere => relation::resolve_interfere(&wave_a, &wave_b)?,
            ResolveStrategy::Branch => relation::resolve_branch(&wave_a, &wave_b)?,
            ResolveStrategy::Rebase(target_name) => {
                let target = self.get_resolve_wave(target_name)?;
                relation::resolve_rebase(&wave_a, &target)?
            }
            ResolveStrategy::Superpose => relation::resolve_superpose(&wave_a, &wave_b)?,
        };
        let elapsed = start.elapsed();

        let probs = resolved.gaze();
        let top: Vec<String> = probs.iter().enumerate()
            .take(5)
            .map(|(i, p)| format!("[{}]={:.4}", i, p))
            .collect();

        out.push(format!(
            "[resolve] {} {} with {}: t={:.3} ({:.1}us)",
            cmd.observations.join(", "), strategy_name,
            strategy_name, resolved.t, elapsed.as_secs_f64() * 1e6,
        ));
        out.push(format!("  probs: {}", top.join(", ")));

        // Store the resolved wave under a combined name
        let resolved_name = format!("resolved_{}_{}", cmd.observations[0], cmd.observations[1]);
        self.waves.insert(resolved_name, resolved);

        Ok(())
    }

    /// Helper: look up a wave from observations, relations, or waves map.
    fn get_resolve_wave(&self, name: &str) -> Result<Wave> {
        if let Some((wave, _)) = self.observations.get(name) {
            return Ok(wave.clone());
        }
        if let Some(rel) = self.relations.get(name) {
            return Ok(rel.wave.clone());
        }
        if let Some(wave) = self.waves.get(name) {
            return Ok(wave.clone());
        }
        Err(AxolError::Relation(format!(
            "No observation, relation, or wave named '{}' found", name
        )))
    }

    fn print_observation_details(&self, obs: &Observation, out: &mut Vec<String>) {
        // Probabilities (top 5)
        let mut indexed: Vec<(usize, f32)> = obs.probabilities.data.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = indexed.iter().take(5)
            .map(|(i, p)| format!("[{}]={:.4}", i, p))
            .collect();
        out.push(format!("  probs: {}", top.join(", ")));

        // Quantum metrics
        if let Some(q_phi) = obs.quantum_phi {
            out.push(format!("  quantum: phi={:.4} omega={:.4}",
                q_phi, obs.quantum_omega.unwrap_or(0.0)));
        }
        if let Some(ref dm) = obs.density_matrix {
            out.push(format!("  density: purity={:.4}", dm.purity()));
        }
    }
}
