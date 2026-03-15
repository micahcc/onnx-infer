pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod inference;

pub use inference::InferenceEngine;
pub use inference::InferenceError;
pub use inference::Tensor;
