//! AXOL DSL Compiler — executes parsed AST through the Declare/Weave/Observe pipeline.

use std::collections::HashMap;
use std::time::Instant;

use crate::types::FloatVec;
use crate::declare::*;
use crate::weaver::{self, Tapestry};
use crate::observatory::{self, Observation};
use crate::compose;
use crate::learn;
use crate::dsl::parser::*;
use crate::errors::{AxolError, Result};

pub struct Runtime {
    declarations: HashMap<String, EntangleDeclaration>,
    tapestries: HashMap<String, Tapestry>,
    chains: HashMap<String, compose::TapestryChain>,
    basin_designs: HashMap<String, compose::BasinDesign>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            tapestries: HashMap::new(),
            chains: HashMap::new(),
            basin_designs: HashMap::new(),
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
        let tapestry = weaver::weave(&decl, cmd.quantum, cmd.seed)?;
        let elapsed = start.elapsed();

        out.push(format!(
            "[weave] '{}': quantum={} seed={} omega={:.4} phi={:.4} ({:.3}ms)",
            cmd.name, cmd.quantum, cmd.seed,
            tapestry.report.estimated_omega,
            tapestry.report.estimated_phi,
            elapsed.as_secs_f64() * 1000.0,
        ));

        if let Some(ref dm) = tapestry.density_matrix {
            out.push(format!("  density: purity={:.4} dim={}", dm.purity(), dm.dim));
        }

        self.tapestries.insert(cmd.name.clone(), tapestry);
        Ok(())
    }

    fn exec_observe(&mut self, cmd: &ObserveCmd, out: &mut Vec<String>) -> Result<()> {
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
