#[derive(Debug)]
pub enum InferenceError {
    ParseError(prost::DecodeError),
    UnsupportedOperator(String),
    InvalidModel(String),
    ShapeMismatch(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::ParseError(e) => write!(f, "Parse error: {e}"),
            InferenceError::UnsupportedOperator(op) => write!(f, "Unsupported operator: {op}"),
            InferenceError::InvalidModel(msg) => write!(f, "Invalid model: {msg}"),
            InferenceError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for InferenceError {}

pub type Result<T> = std::result::Result<T, InferenceError>;
