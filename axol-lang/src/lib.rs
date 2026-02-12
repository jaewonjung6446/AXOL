//! AXOL — Chaos-theory based Declare / Weave / Observe programming language.
//!
//! Core mapping:
//!   - Tapestry = Strange Attractor
//!   - Omega (Cohesion) = 1/(1+max(lambda,0))  (Lyapunov inverse)
//!   - Phi (Clarity) = 1/(1+D/D_max)           (Fractal dim inverse)

pub mod errors;
pub mod types;
pub mod ops;
pub mod density;
pub mod declare;
pub mod dynamics;
pub mod weaver;
pub mod observatory;
pub mod compose;
pub mod codegen;
pub mod axol_ai;
pub mod learn;
pub mod dsl;
