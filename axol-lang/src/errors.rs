//! AXOL error types.

use std::fmt;

#[derive(Debug, Clone)]
pub enum AxolError {
    Quantum(String),
    Weaver(String),
    Observatory(String),
    Parse(String),
    DimensionMismatch { expected: usize, got: usize },
    InvalidInput(String),
    Compose(String),
}

impl fmt::Display for AxolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quantum(msg) => write!(f, "QuantumError: {msg}"),
            Self::Weaver(msg) => write!(f, "WeaverError: {msg}"),
            Self::Observatory(msg) => write!(f, "ObservatoryError: {msg}"),
            Self::Parse(msg) => write!(f, "ParseError: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "DimensionMismatch: expected {expected}, got {got}")
            }
            Self::InvalidInput(msg) => write!(f, "InvalidInput: {msg}"),
            Self::Compose(msg) => write!(f, "ComposeError: {msg}"),
        }
    }
}

impl std::error::Error for AxolError {}

pub type Result<T> = std::result::Result<T, AxolError>;
