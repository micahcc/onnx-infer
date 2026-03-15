pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod layers;

use std::collections::HashMap;

use prost::Message;

use crate::layers::PlanNode;
use crate::layers::build_plan;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float,
    Int64,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub dtype: DType,
    f32_buf: Vec<f32>,
    i64_buf: Vec<i64>,
}

impl Tensor {
    pub fn new(dims: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            dims,
            dtype: DType::Float,
            f32_buf: data,
            i64_buf: vec![],
        }
    }

    pub fn new_i64(dims: Vec<usize>, data: Vec<i64>) -> Self {
        Self {
            dims,
            dtype: DType::Int64,
            f32_buf: vec![],
            i64_buf: data,
        }
    }

    pub fn floats(&self) -> &[f32] {
        assert!(
            self.dtype == DType::Float,
            "expected float tensor, got int64"
        );
        &self.f32_buf
    }

    pub fn ints(&self) -> &[i64] {
        assert!(
            self.dtype == DType::Int64,
            "expected int64 tensor, got float"
        );
        &self.i64_buf
    }

    pub fn into_f32_vec(self) -> Vec<f32> {
        match self.dtype {
            DType::Float => self.f32_buf,
            DType::Int64 => self.i64_buf.iter().map(|&v| v as f32).collect(),
        }
    }

    pub fn into_i64_vec(self) -> Vec<i64> {
        match self.dtype {
            DType::Float => self.f32_buf.iter().map(|&v| v as i64).collect(),
            DType::Int64 => self.i64_buf,
        }
    }

    pub fn f32_at(&self, idx: usize) -> f32 {
        match self.dtype {
            DType::Float => self.f32_buf[idx],
            DType::Int64 => self.i64_buf[idx] as f32,
        }
    }

    pub fn i64_at(&self, idx: usize) -> i64 {
        match self.dtype {
            DType::Float => self.f32_buf[idx] as i64,
            DType::Int64 => self.i64_buf[idx],
        }
    }

    pub fn from_proto(proto: &TensorProto) -> Result<Self> {
        let dims: Vec<usize> = proto.dims.iter().map(|&d| d as usize).collect();
        if proto.data_type == 6 || proto.data_type == 7 {
            let data = extract_int_data(proto)?;
            Ok(Self::new_i64(dims, data))
        } else {
            let data = extract_float_data(proto)?;
            Ok(Self::new(dims, data))
        }
    }

    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self> {
        let proto = TensorProto::decode(bytes).map_err(InferenceError::ParseError)?;
        Self::from_proto(&proto)
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn as_mut_f32(&mut self, len: usize) -> &mut Vec<f32> {
        self.dtype = DType::Float;
        self.f32_buf.resize(len, 0.0);
        &mut self.f32_buf
    }

    pub fn as_mut_i64(&mut self, len: usize) -> &mut Vec<i64> {
        self.dtype = DType::Int64;
        self.i64_buf.resize(len, 0);
        &mut self.i64_buf
    }

    pub fn data_replace_f32(&mut self, data: Vec<f32>) {
        self.dtype = DType::Float;
        self.f32_buf = data;
    }

    pub fn copy_from(&mut self, other: &Tensor) {
        self.dims.clone_from(&other.dims);
        self.dtype = other.dtype;
        match other.dtype {
            DType::Float => {
                self.f32_buf.clear();
                self.f32_buf.extend_from_slice(&other.f32_buf);
            }
            DType::Int64 => {
                self.i64_buf.clear();
                self.i64_buf.extend_from_slice(&other.i64_buf);
            }
        }
    }

    /// Copy data from another tensor, casting to f32. Reuses existing buffer.
    pub fn copy_cast_f32(&mut self, src: &Tensor) {
        self.dims.clone_from(&src.dims);
        self.dtype = DType::Float;
        let len = src.numel();
        self.f32_buf.resize(len, 0.0);
        match src.dtype {
            DType::Float => {
                self.f32_buf[..len].copy_from_slice(&src.f32_buf[..len]);
            }
            DType::Int64 => {
                for i in 0..len {
                    self.f32_buf[i] = src.i64_buf[i] as f32;
                }
            }
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            dims: vec![],
            dtype: DType::Float,
            f32_buf: vec![],
            i64_buf: vec![],
        }
    }
}

fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
    let dtype = tensor.data_type;

    if !tensor.raw_data.is_empty() {
        return match dtype {
            2 => Ok(tensor.raw_data.iter().map(|&b| b as f32).collect()),
            3 => Ok(tensor.raw_data.iter().map(|&b| (b as i8) as f32).collect()),
            11 => Ok(tensor
                .raw_data
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect()),
            _ => Ok(tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
        };
    }
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(tensor.int32_data.iter().map(|&v| v as f32).collect());
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor.int64_data.iter().map(|&v| v as f32).collect());
    }
    Ok(vec![])
}

fn extract_int_data(tensor: &TensorProto) -> Result<Vec<i64>> {
    if !tensor.raw_data.is_empty() {
        return match tensor.data_type {
            6 => Ok(tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64)
                .collect()),
            7 => Ok(tensor
                .raw_data
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect()),
            _ => unreachable!(),
        };
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor.int64_data.clone());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(tensor.int32_data.iter().map(|&v| v as i64).collect());
    }
    Ok(vec![])
}

pub struct InferenceEngine {
    plan: Vec<PlanNode>,
    initializers: HashMap<String, Tensor>,
    output_names: Vec<String>,
    input_sizes: HashMap<String, Vec<usize>>,
}

impl InferenceEngine {
    pub fn new(model_bytes: &[u8], input_sizes: HashMap<String, Vec<usize>>) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).map_err(InferenceError::ParseError)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| InferenceError::InvalidModel("Model has no graph".into()))?;

        let (plan, initializers, output_names) = build_plan(graph)?;

        Ok(Self {
            plan,
            initializers,
            output_names,
            input_sizes,
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let output_names = self.output_names.clone();
        self.run_with_outputs(inputs, &output_names)
    }

    pub fn run_with_outputs(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let _span = tracing::trace_span!("inference").entered();

        let mut values: HashMap<String, Tensor> = inputs;

        // Load initializers
        for (k, v) in &self.initializers {
            values.insert(k.clone(), v.clone());
        }

        // Execute plan
        for node in &mut self.plan {
            match node {
                PlanNode::Single { output, layer } => {
                    if output.is_empty() {
                        continue;
                    }
                    let _span = tracing::trace_span!("op").entered();
                    let mut out = values.remove(output.as_str()).unwrap_or_default();
                    let result = layer.execute(&values, &mut out);
                    values.insert(output.clone(), out);
                    result?;
                }
                PlanNode::Loop(loop_layer) => {
                    loop_layer.execute(&mut values)?;
                }
            }
        }

        // Collect requested outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            if let Some(tensor) = values.get(name) {
                outputs.insert(name.clone(), tensor.clone());
            }
        }

        Ok(outputs)
    }

    pub fn input_sizes(&self) -> &HashMap<String, Vec<usize>> {
        &self.input_sizes
    }
}

// --- Shared helpers used by layer submodules ---

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

pub fn get_tensor<'a>(values: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    values
        .get(name)
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
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    fn setup_tracing(
        test_name: &str,
    ) -> (
        tracing_chrome::FlushGuard,
        tracing::subscriber::DefaultGuard,
    ) {
        let trace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("traces");
        fs::create_dir_all(&trace_dir).ok();
        let trace_path = trace_dir.join(format!("{test_name}.json"));
        let (chrome_layer, flush_guard) = ChromeLayerBuilder::new()
            .file(trace_path)
            .include_args(true)
            .build();
        let subscriber = tracing_subscriber::registry().with(chrome_layer);
        let default_guard = tracing::subscriber::set_default(subscriber);
        (flush_guard, default_guard)
    }

    fn load_model_and_inputs(
        base: &Path,
        model_file: &str,
        test_set: usize,
    ) -> (
        Vec<u8>,
        HashMap<String, Tensor>,
        HashMap<String, Vec<usize>>,
    ) {
        let model_bytes = fs::read(base.join(model_file)).expect("read model");
        let model = ModelProto::decode(&model_bytes[..]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        let test_dir = base.join(format!("test_data_set_{test_set}"));

        let mut inputs = HashMap::new();
        let mut input_sizes = HashMap::new();
        for i in 0..graph.input.len() {
            let pb_path = test_dir.join(format!("input_{i}.pb"));
            if pb_path.exists() {
                let input = Tensor::from_proto_bytes(&fs::read(&pb_path).expect("read input"))
                    .expect("parse input");
                let name = graph.input[i].name.clone();
                input_sizes.insert(name.clone(), input.dims.clone());
                inputs.insert(name, input);
            }
        }

        (model_bytes, inputs, input_sizes)
    }

    fn run_fixture(base: &Path, model_file: &str, test_set: usize) {
        let (model_bytes, inputs, input_sizes) = load_model_and_inputs(base, model_file, test_set);
        let mut engine = InferenceEngine::new(&model_bytes, input_sizes).expect("load model");

        let model = ModelProto::decode(&model_bytes[..]).unwrap();
        let graph = model.graph.as_ref().unwrap();
        let output_name = graph.output[0].name.clone();

        let test_dir = base.join(format!("test_data_set_{test_set}"));
        let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
        let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

        let outputs = engine.run(inputs).expect("inference");
        let output = &outputs[&output_name];

        assert_eq!(output.dims, expected.dims);

        let out_data = output.floats();
        let exp_data = expected.floats();
        for (got, want) in out_data.iter().zip(exp_data.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-3, epsilon = 1e-5);
        }

        let got_class = out_data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let expected_class = exp_data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(got_class, expected_class);
    }

    fn run_quantized_fixture(base: &Path, model_file: &str, test_set: usize) {
        let (model_bytes, inputs, input_sizes) = load_model_and_inputs(base, model_file, test_set);
        let mut engine = InferenceEngine::new(&model_bytes, input_sizes).expect("load model");

        let model = ModelProto::decode(&model_bytes[..]).unwrap();
        let graph = model.graph.as_ref().unwrap();
        let output_name = graph.output[0].name.clone();

        let test_dir = base.join(format!("test_data_set_{test_set}"));
        let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
        let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

        let outputs = engine.run(inputs).expect("inference");
        let output = &outputs[&output_name];

        assert_eq!(output.dims, expected.dims);

        fn softmax(logits: &[f32]) -> Vec<f32> {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|&e| e / sum).collect()
        }

        let got_probs = softmax(output.floats());
        let want_probs = softmax(expected.floats());

        let mut max_abs_err: f32 = 0.0;
        for (g, w) in got_probs.iter().zip(want_probs.iter()) {
            max_abs_err = max_abs_err.max((g - w).abs());
        }
        eprintln!("int8 max softmax probability error: {max_abs_err:.6}");
        assert!(max_abs_err < 0.1, "max softmax error {max_abs_err} >= 0.1");

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

    fn run_multi_io_fixture(base: &Path, model_file: &str, test_set: usize) {
        let (model_bytes, inputs, input_sizes) = load_model_and_inputs(base, model_file, test_set);
        let mut engine = InferenceEngine::new(&model_bytes, input_sizes).expect("load model");

        let model = ModelProto::decode(&model_bytes[..]).unwrap();
        let graph = model.graph.as_ref().unwrap();

        let test_dir = base.join(format!("test_data_set_{test_set}"));
        let outputs = engine.run(inputs).expect("inference");

        for i in 0..graph.output.len() {
            let pb_path = test_dir.join(format!("output_{i}.pb"));
            if pb_path.exists() {
                let expected = Tensor::from_proto_bytes(&fs::read(&pb_path).expect("read output"))
                    .expect("parse output");
                let name = &graph.output[i].name;
                let output = outputs
                    .get(name)
                    .unwrap_or_else(|| panic!("missing output {name}"));
                assert_eq!(output.dims, expected.dims, "shape mismatch for {name}");

                match output.dtype {
                    DType::Float => {
                        let got = output.floats();
                        let want = expected.floats();
                        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                            assert!(
                                (g - w).abs() < 1e-3 || (g - w).abs() / w.abs().max(1e-6) < 1e-3,
                                "output {name}[{j}]: got {g}, want {w}"
                            );
                        }
                    }
                    DType::Int64 => {
                        let got = output.ints();
                        let want = expected.ints();
                        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                            assert_eq!(g, w, "output {name}[{j}]: got {g}, want {w}");
                        }
                    }
                }
            }
        }
    }

    // --- MNIST models ---

    #[test]
    fn test_mnist1_set_0() {
        let _t = setup_tracing("mnist1_set_0");
        run_fixture(&fixture("mnist-1"), "model.onnx", 0);
    }

    #[test]
    fn test_mnist1_set_1() {
        let _t = setup_tracing("mnist1_set_1");
        run_fixture(&fixture("mnist-1"), "model.onnx", 1);
    }

    #[test]
    fn test_mnist1_set_2() {
        let _t = setup_tracing("mnist1_set_2");
        run_fixture(&fixture("mnist-1"), "model.onnx", 2);
    }

    #[test]
    fn test_mnist7_set_0() {
        let _t = setup_tracing("mnist7_set_0");
        run_fixture(&fixture("mnist-7"), "model.onnx", 0);
    }

    #[test]
    fn test_mnist8_set_0() {
        let _t = setup_tracing("mnist8_set_0");
        run_fixture(&fixture("mnist-8"), "model.onnx", 0);
    }

    #[test]
    fn test_mnist12_set_0() {
        let _t = setup_tracing("mnist12_set_0");
        run_fixture(&fixture("mnist-12"), "mnist-12.onnx", 0);
    }

    #[test]
    fn test_mnist12_int8_set_0() {
        let _t = setup_tracing("mnist12_int8_set_0");
        run_quantized_fixture(&fixture("mnist-12-int8"), "mnist-12-int8.onnx", 0);
    }

    // --- MobileNetV2 models ---

    #[test]
    fn test_mobilenetv2_7_set_0() {
        let _t = setup_tracing("mobilenetv2_7_set_0");
        run_fixture(&fixture("mobilenetv2-7"), "mobilenetv2-7.onnx", 0);
    }

    #[test]
    fn test_mobilenetv2_12_set_0() {
        let _t = setup_tracing("mobilenetv2_12_set_0");
        run_fixture(&fixture("mobilenetv2-12"), "mobilenetv2-12.onnx", 0);
    }

    #[test]
    fn test_mobilenetv2_12_int8_set_0() {
        let _t = setup_tracing("mobilenetv2_12_int8_set_0");
        run_quantized_fixture(
            &fixture("mobilenetv2-12-int8"),
            "mobilenetv2-12-int8.onnx",
            0,
        );
    }

    #[test]
    fn test_mobilenetv2_12_qdq_set_0() {
        let _t = setup_tracing("mobilenetv2_12_qdq_set_0");
        run_quantized_fixture(&fixture("mobilenetv2-12-qdq"), "mobilenetv2-12-qdq.onnx", 0);
    }

    // --- Tiny YOLOv2 models ---

    #[test]
    fn test_tinyyolov2_7_set_0() {
        let _t = setup_tracing("tinyyolov2_7_set_0");
        run_fixture(&fixture("tinyyolov2-7"), "model.onnx", 0);
    }

    #[test]
    fn test_tinyyolov2_8_set_0() {
        let _t = setup_tracing("tinyyolov2_8_set_0");
        run_fixture(&fixture("tinyyolov2-8"), "model.onnx", 0);
    }

    // --- Tiny YOLOv3 models ---

    #[test]
    fn test_tinyyolov3_11_set_0() {
        let _t = setup_tracing("tinyyolov3_11_set_0");
        run_multi_io_fixture(&fixture("tiny-yolov3-11"), "yolov3-tiny.onnx", 0);
    }
}
