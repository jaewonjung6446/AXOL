//! AXOL — Chaos-theory based Declare / Weave / Observe programming language.
//!
//! Core mapping:
//!   - Tapestry = Strange Attractor + BasinStructure
//!   - Omega (Cohesion) = 1 - H/ln(k)  (Shannon entropy of basin volumes)
//!   - Phi (Clarity) = 1/(1+D/D_max)   (Fractal dim inverse)
//!   - Theory layer: time-independent (BasinStructure: geometry + probability)
//!   - Implementation layer: time-dependent (ChaosEngine: dynamics → basins)

pub mod errors;
pub mod types;
pub mod ops;
pub mod density;
pub mod collapse;
pub mod wave;
pub mod declare;
pub mod dynamics;
pub mod weaver;
pub mod observatory;
pub mod compose;
pub mod codegen;
pub mod axol_ai;
pub mod learn;
pub mod relation;
pub mod dsl;
