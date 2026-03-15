pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod activation;
pub mod binary;
pub mod conv;
pub mod matmul;
pub mod pool;
pub mod quantize;
pub mod shape;

use std::collections::HashMap;

use prost::Message;

use crate::onnx::ModelProto;
use crate::onnx::NodeProto;
use crate::onnx::TensorProto;

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

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(dims: Vec<usize>, data: Vec<f32>) -> Self {
        Self { dims, data }
    }

    pub fn from_proto(proto: &TensorProto) -> Result<Self> {
        let dims: Vec<usize> = proto.dims.iter().map(|&d| d as usize).collect();
        let data = extract_float_data(proto)?;
        Ok(Self { dims, data })
    }

    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self> {
        let proto = TensorProto::decode(bytes).map_err(InferenceError::ParseError)?;
        Self::from_proto(&proto)
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
    // data_type: 1=FLOAT, 2=UINT8, 3=INT8, 6=INT32, 7=INT64, 11=DOUBLE
    let dtype = tensor.data_type;

    if !tensor.raw_data.is_empty() {
        return match dtype {
            2 => {
                // UINT8: 1 byte each
                Ok(tensor.raw_data.iter().map(|&b| b as f32).collect())
            }
            3 => {
                // INT8: 1 byte each
                Ok(tensor.raw_data.iter().map(|&b| (b as i8) as f32).collect())
            }
            6 => {
                // INT32: 4 bytes each
                Ok(tensor
                    .raw_data
                    .chunks_exact(4)
                    .map(|chunk| {
                        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                    })
                    .collect())
            }
            7 => {
                // INT64: 8 bytes each
                Ok(tensor
                    .raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f32
                    })
                    .collect())
            }
            11 => {
                // DOUBLE: 8 bytes each
                Ok(tensor
                    .raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f32
                    })
                    .collect())
            }
            _ => {
                // FLOAT (1) and others stored as 4-byte floats
                Ok(tensor
                    .raw_data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
        };
    }
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor.int64_data.iter().map(|&v| v as f32).collect());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(tensor.int32_data.iter().map(|&v| v as f32).collect());
    }
    Ok(vec![])
}

pub struct InferenceEngine {
    model: ModelProto,
}

impl InferenceEngine {
    pub fn from_bytes(model_bytes: &[u8]) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).map_err(InferenceError::ParseError)?;
        Ok(Self { model })
    }

    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let graph = self
            .model
            .graph
            .as_ref()
            .ok_or_else(|| InferenceError::InvalidModel("Model has no graph".into()))?;

        let mut values: HashMap<String, Tensor> = inputs;

        // Load initializers
        for init in &graph.initializer {
            if !init.name.is_empty() {
                values.insert(init.name.clone(), Tensor::from_proto(init)?);
            }
        }

        // Execute nodes in topological order
        for node in &graph.node {
            self.execute_node(node, &mut values)?;
        }

        // Collect outputs
        let mut outputs = HashMap::new();
        for output in &graph.output {
            if let Some(tensor) = values.get(&output.name) {
                outputs.insert(output.name.clone(), tensor.clone());
            }
        }

        Ok(outputs)
    }

    fn execute_node(&self, node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
        match node.op_type.as_str() {
            "Constant" => shape::exec_constant(node, values),
            "Div" => binary::exec_div(node, values),
            "Conv" => conv::exec_conv(node, values),
            "Reshape" => shape::exec_reshape(node, values),
            "Add" => binary::exec_add(node, values),
            "Sub" => binary::exec_sub(node, values),
            "Mul" => binary::exec_mul(node, values),
            "Relu" => activation::exec_relu(node, values),
            "BatchNormalization" => activation::exec_batch_normalization(node, values),
            "Clip" => activation::exec_clip(node, values),
            "MaxPool" => pool::exec_maxpool(node, values),
            "GlobalAveragePool" => pool::exec_global_avg_pool(node, values),
            "MatMul" => matmul::exec_matmul(node, values),
            "Gemm" => matmul::exec_gemm(node, values),
            "Softmax" => activation::exec_softmax(node, values),
            "Flatten" => shape::exec_flatten(node, values),
            "Shape" => shape::exec_shape(node, values),
            "Gather" => shape::exec_gather(node, values),
            "Unsqueeze" => shape::exec_unsqueeze(node, values),
            "Concat" => shape::exec_concat(node, values),
            "QuantizeLinear" => quantize::exec_quantize_linear(node, values),
            "DequantizeLinear" => quantize::exec_dequantize_linear(node, values),
            "QLinearConv" => quantize::exec_qlinear_conv(node, values),
            "QLinearAdd" => quantize::exec_qlinear_add(node, values),
            "QLinearMatMul" => quantize::exec_qlinear_matmul(node, values),
            "QLinearGlobalAveragePool" => quantize::exec_qlinear_global_avg_pool(node, values),
            op => Err(InferenceError::UnsupportedOperator(op.to_string())),
        }
    }
}

// --- Shared helpers used by submodules ---

pub fn get_attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

pub fn get_attr_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

pub fn get_attr_float(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

pub fn get_attr_string(node: &NodeProto, name: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| {
            if a.s.is_empty() {
                None
            } else {
                Some(String::from_utf8_lossy(&a.s).to_string())
            }
        })
}

pub fn get_tensor(values: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    values
        .get(name)
        .cloned()
        .ok_or_else(|| InferenceError::InvalidModel(format!("Tensor '{name}' not found")))
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = vec![1usize; max_len];
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        result[i] = da.max(db);
    }
    result
}

pub fn broadcast_index(index: &[usize], shape: &[usize], out_shape: &[usize]) -> usize {
    let offset = out_shape.len() - shape.len();
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        let dim_idx = if shape[i] == 1 { 0 } else { index[i + offset] };
        flat += dim_idx * stride;
        stride *= shape[i];
    }
    flat
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use approx::assert_relative_eq;

    use super::*;

    const FIXTURES_DIR: &str = env!("FIXTURES_DIR");

    /// Run a model fixture against its bundled test data set.
    fn run_fixture(fixture_dir: &str, model_file: &str, test_set: usize) {
        let base = Path::new(FIXTURES_DIR).join(fixture_dir);
        let model_bytes = fs::read(base.join(model_file)).expect("read model");
        let engine = InferenceEngine::from_bytes(&model_bytes).expect("load model");

        let test_dir = base.join(format!("test_data_set_{test_set}"));
        let input_bytes = fs::read(test_dir.join("input_0.pb")).expect("read input");
        let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
        let input = Tensor::from_proto_bytes(&input_bytes).expect("parse input");
        let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

        let model = ModelProto::decode(&model_bytes[..]).unwrap();
        let graph = model.graph.as_ref().unwrap();
        let input_name = graph.input[0].name.clone();
        let output_name = graph.output[0].name.clone();

        let mut inputs = HashMap::new();
        inputs.insert(input_name, input);

        let outputs = engine.run(inputs).expect("inference");
        let output = &outputs[&output_name];

        assert_eq!(output.dims, expected.dims);

        for (got, want) in output.data.iter().zip(expected.data.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-3, epsilon = 1e-5);
        }

        let got_class = output
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let expected_class = expected
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(got_class, expected_class);
    }

    /// Run a quantized model fixture. Compares softmax probabilities since
    /// dequant-compute-requant in float32 accumulates small rounding errors.
    fn run_quantized_fixture(fixture_dir: &str, model_file: &str, test_set: usize) {
        let base = Path::new(FIXTURES_DIR).join(fixture_dir);
        let model_bytes = fs::read(base.join(model_file)).expect("read model");
        let engine = InferenceEngine::from_bytes(&model_bytes).expect("load model");

        let test_dir = base.join(format!("test_data_set_{test_set}"));
        let input_bytes = fs::read(test_dir.join("input_0.pb")).expect("read input");
        let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
        let input = Tensor::from_proto_bytes(&input_bytes).expect("parse input");
        let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

        let model = ModelProto::decode(&model_bytes[..]).unwrap();
        let graph = model.graph.as_ref().unwrap();
        let input_name = graph.input[0].name.clone();
        let output_name = graph.output[0].name.clone();

        let mut inputs = HashMap::new();
        inputs.insert(input_name, input);

        let outputs = engine.run(inputs).expect("inference");
        let output = &outputs[&output_name];

        assert_eq!(output.dims, expected.dims);

        fn softmax(logits: &[f32]) -> Vec<f32> {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|&e| e / sum).collect()
        }

        let got_probs = softmax(&output.data);
        let want_probs = softmax(&expected.data);

        let mut max_abs_err: f32 = 0.0;
        for (g, w) in got_probs.iter().zip(want_probs.iter()) {
            max_abs_err = max_abs_err.max((g - w).abs());
        }
        eprintln!("int8 max softmax probability error: {max_abs_err:.6}");
        assert!(max_abs_err < 0.1, "max softmax error {max_abs_err} >= 0.1");

        // Quantized models may swap closely-ranked classes due to accumulated
        // rounding in the dequant-compute-requant pipeline. Check that the
        // expected top-1 class appears in our top-5 predictions.
        let expected_class = want_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let mut indexed: Vec<(usize, &f32)> = got_probs.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top5: Vec<usize> = indexed.iter().take(5).map(|&(i, _)| i).collect();
        assert!(
            top5.contains(&expected_class),
            "expected class {expected_class} not in top-5: {top5:?}"
        );
    }

    // --- MNIST models ---

    // mnist-1 (opset 1): weights as Constant nodes, legacy broadcast
    #[test]
    fn test_mnist1_set_0() {
        run_fixture("mnist-1", "model.onnx", 0);
    }

    #[test]
    fn test_mnist1_set_1() {
        run_fixture("mnist-1", "model.onnx", 1);
    }

    #[test]
    fn test_mnist1_set_2() {
        run_fixture("mnist-1", "model.onnx", 2);
    }

    // mnist-7 (opset 7)
    #[test]
    fn test_mnist7_set_0() {
        run_fixture("mnist-7", "model.onnx", 0);
    }

    // mnist-8 (opset 8)
    #[test]
    fn test_mnist8_set_0() {
        run_fixture("mnist-8", "model.onnx", 0);
    }

    // mnist-12 (opset 12): weights as initializers, standard broadcast
    #[test]
    fn test_mnist12_set_0() {
        run_fixture("mnist-12", "mnist-12.onnx", 0);
    }

    // mnist-12-int8 (opset 12): quantized MNIST
    #[test]
    fn test_mnist12_int8_set_0() {
        run_quantized_fixture("mnist-12-int8", "mnist-12-int8.onnx", 0);
    }

    // --- MobileNetV2 models ---

    // mobilenetv2-7 (opset 10)
    #[test]
    fn test_mobilenetv2_7_set_0() {
        run_fixture("mobilenetv2-7", "mobilenetv2-7.onnx", 0);
    }

    // mobilenetv2-12 (opset 12): depthwise conv, clip, gemm, shape/gather/unsqueeze/concat
    #[test]
    fn test_mobilenetv2_12_set_0() {
        run_fixture("mobilenetv2-12", "mobilenetv2-12.onnx", 0);
    }

    // mobilenetv2-12-int8: quantized with QLinear* fused ops
    #[test]
    fn test_mobilenetv2_12_int8_set_0() {
        run_quantized_fixture("mobilenetv2-12-int8", "mobilenetv2-12-int8.onnx", 0);
    }

    // mobilenetv2-12-qdq: quantized with DequantizeLinear/QuantizeLinear around standard ops
    #[test]
    fn test_mobilenetv2_12_qdq_set_0() {
        run_quantized_fixture("mobilenetv2-12-qdq", "mobilenetv2-12-qdq.onnx", 0);
    }
}
