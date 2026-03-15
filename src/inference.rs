use std::collections::HashMap;

use prost::Message;

use crate::onnx::{ModelProto, NodeProto, TensorProto};

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
        let proto =
            TensorProto::decode(bytes).map_err(InferenceError::ParseError)?;
        Self::from_proto(&proto)
    }

    fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
    if !tensor.raw_data.is_empty() {
        let data: Vec<f32> = tensor
            .raw_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        return Ok(data);
    }
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
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

    fn execute_node(
        &self,
        node: &NodeProto,
        values: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        match node.op_type.as_str() {
            "Constant" => exec_constant(node, values),
            "Div" => exec_div(node, values),
            "Conv" => exec_conv(node, values),
            "Reshape" => exec_reshape(node, values),
            "Add" => exec_add(node, values),
            "Relu" => exec_relu(node, values),
            "MaxPool" => exec_maxpool(node, values),
            "MatMul" => exec_matmul(node, values),
            "Softmax" => exec_softmax(node, values),
            "Flatten" => exec_flatten(node, values),
            op => Err(InferenceError::UnsupportedOperator(op.to_string())),
        }
    }
}

fn get_attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

fn get_attr_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
}

fn get_attr_string(node: &NodeProto, name: &str) -> Option<String> {
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

fn get_tensor(values: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    values
        .get(name)
        .cloned()
        .ok_or_else(|| InferenceError::InvalidModel(format!("Tensor '{name}' not found")))
}

fn exec_constant(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| InferenceError::InvalidModel("Constant has no value".into()))?;

    let tensor_proto = attr
        .t
        .as_ref()
        .ok_or_else(|| InferenceError::InvalidModel("Constant value is not a tensor".into()))?;

    let tensor = Tensor::from_proto(tensor_proto)?;
    values.insert(node.output[0].clone(), tensor);
    Ok(())
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = vec![1usize; max_len];
    for i in 0..max_len {
        let da = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
        let db = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
        result[i] = da.max(db);
    }
    result
}

fn broadcast_index(index: &[usize], shape: &[usize], out_shape: &[usize]) -> usize {
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

fn exec_binary_op(
    node: &NodeProto,
    values: &mut HashMap<String, Tensor>,
    op: fn(f32, f32) -> f32,
) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let mut b = get_tensor(values, &node.input[1])?;

    // Handle legacy ONNX broadcast attribute: when broadcast=1 and axis is set,
    // the second operand's shape is aligned starting at the given axis.
    // e.g. a=[1,8,24,24], b=[8], axis=1 => reshape b to [1,8,1,1]
    let legacy_broadcast = get_attr_int(node, "broadcast").unwrap_or(0);
    if legacy_broadcast == 1 && b.dims.len() < a.dims.len() {
        let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
        let mut new_dims = vec![1usize; a.dims.len()];
        for (i, &d) in b.dims.iter().enumerate() {
            new_dims[axis + i] = d;
        }
        b.dims = new_dims;
    }

    let out_shape = broadcast_shape(&a.dims, &b.dims);
    let numel: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; numel];

    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];

    for i in 0..numel {
        let ai = broadcast_index(&index, &a.dims, &out_shape);
        let bi = broadcast_index(&index, &b.dims, &out_shape);
        data[i] = op(a.data[ai], b.data[bi]);

        // Increment index
        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(out_shape, data));
    Ok(())
}

fn exec_div(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a / b)
}

fn exec_add(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a + b)
}

fn exec_conv(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let weight = get_tensor(values, &node.input[1])?;

    let bias = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(get_tensor(values, &node.input[2])?)
    } else {
        None
    };

    let kernel_shape = get_attr_ints(node, "kernel_shape").unwrap_or_else(|| {
        vec![weight.dims[2] as i64, weight.dims[3] as i64]
    });
    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let mut pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let dilations = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
    let group = get_attr_int(node, "group").unwrap_or(1) as usize;
    let auto_pad = get_attr_string(node, "auto_pad").unwrap_or_default();

    // input: [N, C, H, W]
    // weight: [M, C/group, kH, kW]
    let n = input.dims[0];
    let c_in = input.dims[1];
    let h_in = input.dims[2];
    let w_in = input.dims[3];

    let c_out = weight.dims[0];
    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let dh = dilations[0] as usize;
    let dw = dilations[1] as usize;

    // Handle auto_pad
    if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
        let oh = (h_in + sh - 1) / sh;
        let ow = (w_in + sw - 1) / sw;
        let pad_h = ((oh - 1) * sh + dh * (kh - 1) + 1).saturating_sub(h_in);
        let pad_w = ((ow - 1) * sw + dw * (kw - 1) + 1).saturating_sub(w_in);
        if auto_pad == "SAME_UPPER" {
            pads = vec![
                (pad_h / 2) as i64,
                (pad_w / 2) as i64,
                (pad_h - pad_h / 2) as i64,
                (pad_w - pad_w / 2) as i64,
            ];
        } else {
            pads = vec![
                (pad_h - pad_h / 2) as i64,
                (pad_w - pad_w / 2) as i64,
                (pad_h / 2) as i64,
                (pad_w / 2) as i64,
            ];
        }
    }

    let ph_begin = pads[0] as usize;
    let pw_begin = pads[1] as usize;

    let h_out = (h_in + pads[0] as usize + pads[2] as usize - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + pads[1] as usize + pads[3] as usize - dw * (kw - 1) - 1) / sw + 1;

    let c_in_per_group = c_in / group;
    let c_out_per_group = c_out / group;

    let mut output = vec![0.0f32; n * c_out * h_out * w_out];

    for batch in 0..n {
        for g in 0..group {
            for oc in 0..c_out_per_group {
                let abs_oc = g * c_out_per_group + oc;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ic in 0..c_in_per_group {
                            let abs_ic = g * c_in_per_group + ic;
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let ih = oh * sh + fh * dh;
                                    let iw = ow * sw + fw * dw;
                                    if ih >= ph_begin
                                        && iw >= pw_begin
                                        && ih - ph_begin < h_in
                                        && iw - pw_begin < w_in
                                    {
                                        let ih = ih - ph_begin;
                                        let iw = iw - pw_begin;
                                        let input_idx =
                                            ((batch * c_in + abs_ic) * h_in + ih) * w_in + iw;
                                        let weight_idx =
                                            ((abs_oc * c_in_per_group + ic) * kh + fh) * kw + fw;
                                        sum += input.data[input_idx] * weight.data[weight_idx];
                                    }
                                }
                            }
                        }
                        if let Some(ref bias) = bias {
                            sum += bias.data[abs_oc];
                        }
                        let out_idx = ((batch * c_out + abs_oc) * h_out + oh) * w_out + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    values.insert(
        node.output[0].clone(),
        Tensor::new(vec![n, c_out, h_out, w_out], output),
    );
    Ok(())
}

fn exec_reshape(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    // Shape can come from attribute or second input
    let new_shape: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let shape_tensor = get_tensor(values, &node.input[1])?;
        shape_tensor.data.iter().map(|&v| v as i64).collect()
    } else {
        // Check for shape attribute (older ONNX format)
        get_attr_ints(node, "shape").ok_or_else(|| {
            InferenceError::InvalidModel("Reshape: no shape input or attribute".into())
        })?
    };

    let total = input.numel();
    let mut dims: Vec<usize> = Vec::new();
    let mut infer_idx: Option<usize> = None;

    for (i, &s) in new_shape.iter().enumerate() {
        if s == -1 {
            infer_idx = Some(i);
            dims.push(0); // placeholder
        } else if s == 0 {
            dims.push(input.dims[i]);
        } else {
            dims.push(s as usize);
        }
    }

    if let Some(idx) = infer_idx {
        let known: usize = dims.iter().enumerate().filter(|&(i, _)| i != idx).map(|(_, &v)| v).product();
        dims[idx] = total / known;
    }

    values.insert(node.output[0].clone(), Tensor::new(dims, input.data));
    Ok(())
}

fn exec_relu(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let data: Vec<f32> = input.data.iter().map(|&v| v.max(0.0)).collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

fn exec_maxpool(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    let kernel_shape = get_attr_ints(node, "kernel_shape")
        .ok_or_else(|| InferenceError::InvalidModel("MaxPool missing kernel_shape".into()))?;
    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let n = input.dims[0];
    let c = input.dims[1];
    let h_in = input.dims[2];
    let w_in = input.dims[3];

    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let ph_begin = pads[0] as usize;
    let pw_begin = pads[1] as usize;

    let h_out = (h_in + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
    let w_out = (w_in + pads[1] as usize + pads[3] as usize - kw) / sw + 1;

    let mut output = vec![f32::NEG_INFINITY; n * c * h_out * w_out];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * sh + fh;
                            let iw = ow * sw + fw;
                            if ih >= ph_begin
                                && iw >= pw_begin
                                && ih - ph_begin < h_in
                                && iw - pw_begin < w_in
                            {
                                let ih = ih - ph_begin;
                                let iw = iw - pw_begin;
                                let idx = ((batch * c + ch) * h_in + ih) * w_in + iw;
                                max_val = max_val.max(input.data[idx]);
                            }
                        }
                    }
                    let out_idx = ((batch * c + ch) * h_out + oh) * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    values.insert(
        node.output[0].clone(),
        Tensor::new(vec![n, c, h_out, w_out], output),
    );
    Ok(())
}

fn exec_matmul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let b = get_tensor(values, &node.input[1])?;

    // Support 2D matmul: [M, K] x [K, N] -> [M, N]
    // Also handle batched: [..., M, K] x [..., K, N] -> [..., M, N]
    let a_rank = a.dims.len();
    let b_rank = b.dims.len();

    let m = a.dims[a_rank - 2];
    let k = a.dims[a_rank - 1];
    let n = b.dims[b_rank - 1];

    // For simplicity, handle 2D case and batch prefix
    let batch_dims_a = &a.dims[..a_rank - 2];
    let batch_dims_b = &b.dims[..b_rank - 2];
    let batch_shape = broadcast_shape(batch_dims_a, batch_dims_b);
    let batch_size: usize = batch_shape.iter().product();

    let mut out_dims = batch_shape.clone();
    out_dims.push(m);
    out_dims.push(n);

    let mut output = vec![0.0f32; batch_size * m * n];

    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let o_batch_stride = m * n;

    for batch in 0..batch_size {
        let a_off = if batch_dims_a.is_empty() || batch_dims_a.iter().product::<usize>() == 1 {
            0
        } else {
            batch * a_batch_stride
        };
        let b_off = if batch_dims_b.is_empty() || batch_dims_b.iter().product::<usize>() == 1 {
            0
        } else {
            batch * b_batch_stride
        };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a.data[a_off + i * k + p] * b.data[b_off + p * n + j];
                }
                output[batch * o_batch_stride + i * n + j] = sum;
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(out_dims, output));
    Ok(())
}

fn exec_softmax(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(-1);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 { (rank + axis) as usize } else { axis as usize };

    let outer: usize = input.dims[..axis].iter().product();
    let dim = input.dims[axis];
    let inner: usize = input.dims[axis + 1..].iter().product();

    let mut data = input.data.clone();

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                max_val = max_val.max(data[idx]);
            }
            let mut sum = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                data[idx] = (data[idx] - max_val).exp();
                sum += data[idx];
            }
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                data[idx] /= sum;
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

fn exec_flatten(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;

    let outer: usize = input.dims[..axis].iter().product();
    let inner: usize = input.dims[axis..].iter().product();

    values.insert(
        node.output[0].clone(),
        Tensor::new(vec![outer, inner], input.data),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    const FIXTURES_DIR: &str = env!("FIXTURES_DIR");

    fn load_mnist_test_set(set_idx: usize) -> (Tensor, Tensor) {
        let dir = Path::new(FIXTURES_DIR)
            .join("mnist")
            .join(format!("test_data_set_{set_idx}"));

        let input_bytes = fs::read(dir.join("input_0.pb")).expect("read input");
        let output_bytes = fs::read(dir.join("output_0.pb")).expect("read output");

        let input = Tensor::from_proto_bytes(&input_bytes).expect("parse input");
        let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

        (input, expected)
    }

    fn run_mnist(set_idx: usize) {
        let model_path = Path::new(FIXTURES_DIR).join("mnist").join("model.onnx");
        let model_bytes = fs::read(&model_path).expect("read model");
        let engine = InferenceEngine::from_bytes(&model_bytes).expect("load model");

        let (input, expected) = load_mnist_test_set(set_idx);

        let mut inputs = HashMap::new();
        inputs.insert("Input73".to_string(), input);

        let outputs = engine.run(inputs).expect("inference");
        let output = &outputs["Plus422_Output_0"];

        assert_eq!(output.dims, expected.dims);

        // Check that values match within tolerance
        for (i, (got, want)) in output.data.iter().zip(expected.data.iter()).enumerate() {
            let diff = (got - want).abs();
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                diff < tol,
                "output[{i}]: got {got}, expected {want}, diff {diff}"
            );
        }

        // Check argmax matches
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

    #[test]
    fn test_mnist_set_0() {
        run_mnist(0);
    }

    #[test]
    fn test_mnist_set_1() {
        run_mnist(1);
    }

    #[test]
    fn test_mnist_set_2() {
        run_mnist(2);
    }
}
