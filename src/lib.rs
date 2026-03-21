//! # onnx-infer
//!
//! Pure Rust ONNX inference engine targeting computer vision models. Zero heap
//! allocations after warmup — all tensor buffers, intermediate values, and output
//! maps are pre-allocated and reused across forward passes.
//!
//! ## Quick start
//!
//! ```no_run
//! use std::collections::HashMap;
//! use onnx_infer::{InferenceEngine, Tensor, dims};
//!
//! let model_bytes = std::fs::read("model.onnx").unwrap();
//! let mut engine = InferenceEngine::new(&model_bytes).unwrap();
//!
//! let input = Tensor::new(dims![1, 1, 28, 28], vec![0.0; 784]);
//! let mut inputs = HashMap::new();
//! inputs.insert("Input3".to_string(), input);
//!
//! engine.run(inputs).unwrap();
//!
//! let output = &engine.outputs["Plus214_Output_0"];
//! println!("dims: {:?}", output.dims);
//! println!("data: {:?}", &output.floats().unwrap()[..10]);
//! ```
//!
//! ## Running inference with external output buffers
//!
//! If you need to control the output buffer (e.g. to avoid copying), use
//! [`InferenceEngine::run_for`] directly:
//!
//! ```no_run
//! # use std::collections::HashMap;
//! # use onnx_infer::{InferenceEngine, Tensor, dims};
//! # let model_bytes = std::fs::read("model.onnx").unwrap();
//! # let mut engine = InferenceEngine::new(&model_bytes).unwrap();
//! let mut outputs = HashMap::new();
//! let output_names = vec!["Plus214_Output_0".to_string()];
//!
//! let input = Tensor::new(dims![1, 1, 28, 28], vec![0.0; 784]);
//! let mut inputs = HashMap::new();
//! inputs.insert("Input3".to_string(), input);
//!
//! engine.run_for(inputs, &output_names, &mut outputs).unwrap();
//! ```
//!
//! ## Inspecting model shapes
//!
//! After constructing an engine you can query the expected input sizes and
//! the inferred shape map for all intermediate tensors:
//!
//! ```no_run
//! # use std::collections::HashMap;
//! # use onnx_infer::{InferenceEngine, dims};
//! # let model_bytes = std::fs::read("model.onnx").unwrap();
//! # let engine = InferenceEngine::new(&model_bytes).unwrap();
//! println!("shapes: {:?}", engine.shape_map());
//! ```

#![allow(clippy::too_many_arguments, clippy::collapsible_if)]

pub mod onnx {
    #![allow(clippy::doc_overindented_list_items)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod blas;
pub mod dtype;
pub mod graph_opt;
pub mod inference_engine;
pub mod inference_error;
pub mod layers;
pub mod onnx_ir;
pub mod tensor_data;
pub mod utils;
#[cfg(feature = "xnnpack")]
pub mod xnnpack_ffi;
pub use dtype::DType;
pub use dtype::ONNX_DOUBLE;
pub use dtype::ONNX_FLOAT;
pub use dtype::ONNX_INT8;
pub use dtype::ONNX_INT32;
pub use dtype::ONNX_INT64;
pub use dtype::ONNX_STRING;
pub use dtype::ONNX_UINT8;
pub use inference_engine::InferenceEngine;
pub use inference_error::Result;
pub use tensor_data::Dims;
pub use tensor_data::Tensor;
pub use tensor_data::TensorData;
pub use utils::broadcast_index;
pub use utils::broadcast_shape;
pub use utils::broadcast_shape_into;
pub use utils::get_tensor;

#[cfg(test)]
mod tests;
