//! AXOL Wave Text Engine (WTE).
//!
//! A text generation architecture based on AXOL's chaos-theoretic foundations.
//! Uses wave resonance reservoirs for context processing and linear readout
//! for prediction.
//!
//! Architecture:
//!   - SPS: Semantic Phase Space — vocabulary → basins → waves
//!   - Reservoir: Multi-scale wave resonance for context processing
//!   - Readout: Linear readout from reservoir features
//!   - Heads: Task-specific heads (autocomplete, generation, classification, etc.)
//!   - Engine: Unified WaveTextEngine API
//!
//! Key properties:
//!   - Context processing: O(n × dim²) instead of O(n² × dim)
//!   - Self-aware: Ω/Φ per generated token
//!   - Offline capable: ~dim² × 4 bytes model size

pub mod tokenizer;
pub mod sps;
pub mod generator;
pub mod data;
pub mod reservoir;
pub mod readout;
pub mod heads;
pub mod fingerprint;
pub mod engine;
pub mod chat;
