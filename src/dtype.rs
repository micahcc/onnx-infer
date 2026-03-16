#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float,
    Int64,
    String,
}

// ONNX TensorProto.DataType constants
pub const ONNX_FLOAT: i32 = 1;
pub const ONNX_UINT8: i32 = 2;
pub const ONNX_INT8: i32 = 3;
pub const ONNX_INT32: i32 = 6;
pub const ONNX_INT64: i32 = 7;
pub const ONNX_STRING: i32 = 8;
pub const ONNX_DOUBLE: i32 = 11;
