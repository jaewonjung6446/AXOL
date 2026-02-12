//! Compose — abstraction layer for practical AXOL programming.
//!
//! Provides: basin design, tapestry composition, logic gates,
//! confidence voting, and convergence-based iteration.

pub mod confidence;
pub mod tapestry_chain;
pub mod iterate;
pub mod basin_designer;
pub mod logic;

pub use confidence::*;
pub use tapestry_chain::*;
pub use iterate::*;
pub use basin_designer::*;
pub use logic::*;
