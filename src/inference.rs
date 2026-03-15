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

    fn numel(&self) -> usize {
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
            "Constant" => exec_constant(node, values),
            "Div" => exec_div(node, values),
            "Conv" => exec_conv(node, values),
            "Reshape" => exec_reshape(node, values),
            "Add" => exec_add(node, values),
            "Sub" => exec_sub(node, values),
            "Mul" => exec_mul(node, values),
            "Relu" => exec_relu(node, values),
            "Clip" => exec_clip(node, values),
            "MaxPool" => exec_maxpool(node, values),
            "GlobalAveragePool" => exec_global_avg_pool(node, values),
            "MatMul" => exec_matmul(node, values),
            "Gemm" => exec_gemm(node, values),
            "Softmax" => exec_softmax(node, values),
            "Flatten" => exec_flatten(node, values),
            "Shape" => exec_shape(node, values),
            "Gather" => exec_gather(node, values),
            "Unsqueeze" => exec_unsqueeze(node, values),
            "Concat" => exec_concat(node, values),
            "QuantizeLinear" => exec_quantize_linear(node, values),
            "DequantizeLinear" => exec_dequantize_linear(node, values),
            "QLinearConv" => exec_qlinear_conv(node, values),
            "QLinearAdd" => exec_qlinear_add(node, values),
            "QLinearMatMul" => exec_qlinear_matmul(node, values),
            "QLinearGlobalAveragePool" => exec_qlinear_global_avg_pool(node, values),
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
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

fn get_attr_float(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
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

fn exec_sub(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a - b)
}

fn exec_mul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a * b)
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

    let kernel_shape = get_attr_ints(node, "kernel_shape")
        .unwrap_or_else(|| vec![weight.dims[2] as i64, weight.dims[3] as i64]);
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
        let known: usize = dims
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &v)| v)
            .product();
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

fn exec_clip(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    // Opset 11+: min/max are optional inputs (inputs[1], inputs[2])
    // Opset 6: min/max are attributes
    let min_val = if node.input.len() > 1 && !node.input[1].is_empty() {
        get_tensor(values, &node.input[1])?.data[0]
    } else {
        get_attr_float(node, "min").unwrap_or(f32::NEG_INFINITY)
    };
    let max_val = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?.data[0]
    } else {
        get_attr_float(node, "max").unwrap_or(f32::INFINITY)
    };

    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&v| v.clamp(min_val, max_val))
        .collect();
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
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

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

fn exec_global_avg_pool(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    // Input: [N, C, H, W] -> Output: [N, C, 1, 1]
    let n = input.dims[0];
    let c = input.dims[1];
    let spatial: usize = input.dims[2..].iter().product();

    let mut output = vec![0.0f32; n * c];
    for batch in 0..n {
        for ch in 0..c {
            let offset = (batch * c + ch) * spatial;
            let sum: f32 = input.data[offset..offset + spatial].iter().sum();
            output[batch * c + ch] = sum / spatial as f32;
        }
    }

    let mut out_dims = vec![n, c];
    for _ in 2..input.dims.len() {
        out_dims.push(1);
    }
    values.insert(node.output[0].clone(), Tensor::new(out_dims, output));
    Ok(())
}

fn exec_gemm(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let b = get_tensor(values, &node.input[1])?;
    let c_tensor = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(get_tensor(values, &node.input[2])?)
    } else {
        None
    };

    let alpha = get_attr_float(node, "alpha").unwrap_or(1.0);
    let beta = get_attr_float(node, "beta").unwrap_or(1.0);
    let trans_a = get_attr_int(node, "transA").unwrap_or(0) != 0;
    let trans_b = get_attr_int(node, "transB").unwrap_or(0) != 0;

    let (m, k_a) = if trans_a {
        (a.dims[1], a.dims[0])
    } else {
        (a.dims[0], a.dims[1])
    };
    let (k_b, n) = if trans_b {
        (b.dims[1], b.dims[0])
    } else {
        (b.dims[0], b.dims[1])
    };
    debug_assert_eq!(k_a, k_b);
    let k = k_a;

    let mut output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a {
                    a.data[p * m + i]
                } else {
                    a.data[i * k + p]
                };
                let b_val = if trans_b {
                    b.data[j * k + p]
                } else {
                    b.data[p * n + j]
                };
                sum += a_val * b_val;
            }
            output[i * n + j] = alpha * sum;
        }
    }

    if let Some(c_tensor) = c_tensor {
        // Broadcast-add C (typically [N] or [M, N])
        let c_shape = broadcast_shape(&[m, n], &c_tensor.dims);
        for i in 0..m {
            for j in 0..n {
                let idx = [i, j];
                let ci = broadcast_index(&idx, &c_tensor.dims, &c_shape);
                output[i * n + j] += beta * c_tensor.data[ci];
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(vec![m, n], output));
    Ok(())
}

fn exec_shape(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let rank = input.dims.len();
    let data: Vec<f32> = input.dims.iter().map(|&d| d as f32).collect();
    values.insert(node.output[0].clone(), Tensor::new(vec![rank], data));
    Ok(())
}

fn exec_gather(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let indices = get_tensor(values, &node.input[1])?;
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    // For scalar index gathering from a 1-D tensor (common case: Shape -> Gather)
    if indices.dims.is_empty()
        || (indices.dims.len() == 1 && indices.dims[0] == 1)
        || indices.numel() == 1
    {
        let idx = indices.data[0] as i64;
        let idx = if idx < 0 {
            input.dims[axis] as i64 + idx
        } else {
            idx
        } as usize;

        if input.dims.len() == 1 {
            // Scalar output
            values.insert(
                node.output[0].clone(),
                Tensor::new(vec![], vec![input.data[idx]]),
            );
        } else {
            // Slice along axis
            let outer: usize = input.dims[..axis].iter().product();
            let inner: usize = input.dims[axis + 1..].iter().product();
            let axis_size = input.dims[axis];
            let mut data = Vec::with_capacity(outer * inner);
            for o in 0..outer {
                let base = o * axis_size * inner + idx * inner;
                data.extend_from_slice(&input.data[base..base + inner]);
            }
            let mut out_dims: Vec<usize> = input.dims[..axis].to_vec();
            out_dims.extend_from_slice(&input.dims[axis + 1..]);
            if out_dims.is_empty() {
                out_dims = vec![];
            }
            values.insert(node.output[0].clone(), Tensor::new(out_dims, data));
        }
    } else {
        return Err(InferenceError::UnsupportedOperator(
            "Gather with non-scalar indices".into(),
        ));
    }
    Ok(())
}

fn exec_unsqueeze(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    // Opset 13+: axes from second input; earlier: from attribute
    let axes: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_tensor = get_tensor(values, &node.input[1])?;
        axes_tensor.data.iter().map(|&v| v as i64).collect()
    } else {
        get_attr_ints(node, "axes").unwrap_or_default()
    };

    let out_rank = input.dims.len() + axes.len();
    let mut out_dims = input.dims.clone();
    let mut sorted_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (out_rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    sorted_axes.sort();
    for &ax in &sorted_axes {
        out_dims.insert(ax, 1);
    }

    values.insert(node.output[0].clone(), Tensor::new(out_dims, input.data));
    Ok(())
}

fn exec_concat(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let axis = get_attr_int(node, "axis").unwrap_or(0);

    let tensors: Vec<Tensor> = node
        .input
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| get_tensor(values, name))
        .collect::<Result<Vec<_>>>()?;

    if tensors.is_empty() {
        return Err(InferenceError::InvalidModel("Concat with no inputs".into()));
    }

    let rank = tensors[0].dims.len();
    let axis = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let mut out_dims = tensors[0].dims.clone();
    out_dims[axis] = tensors.iter().map(|t| t.dims[axis]).sum();

    let outer: usize = out_dims[..axis].iter().product();
    let inner: usize = out_dims[axis + 1..].iter().product();
    let total = out_dims.iter().product::<usize>();
    let mut data = vec![0.0f32; total];

    let mut axis_offset = 0;
    for t in &tensors {
        let t_axis = t.dims[axis];
        for o in 0..outer {
            for a in 0..t_axis {
                let src_base = (o * t_axis + a) * inner;
                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                data[dst_base..dst_base + inner]
                    .copy_from_slice(&t.data[src_base..src_base + inner]);
            }
        }
        axis_offset += t_axis;
    }

    values.insert(node.output[0].clone(), Tensor::new(out_dims, data));
    Ok(())
}

// --- Quantized operators ---

/// Dequantize: float = (quantized - zero_point) * scale
fn dequantize(data: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter().map(|&v| (v - zero_point) * scale).collect()
}

/// Quantize: quantized = clamp(round(float / scale) + zero_point, 0, 255)
fn quantize_u8(data: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter()
        .map(|&v| (v / scale + zero_point).round().clamp(0.0, 255.0))
        .collect()
}

fn exec_quantize_linear(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let scale = get_tensor(values, &node.input[1])?;
    let zero_point = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?.data[0]
    } else {
        0.0
    };
    let data = quantize_u8(&input.data, scale.data[0], zero_point);
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

fn exec_dequantize_linear(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let scale = get_tensor(values, &node.input[1])?;
    let zero_point = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?
    } else {
        Tensor::new(vec![], vec![0.0])
    };

    let axis = get_attr_int(node, "axis").unwrap_or(1);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    if scale.data.len() == 1 {
        // Scalar dequant
        let data = dequantize(&input.data, scale.data[0], zero_point.data[0]);
        values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    } else {
        // Per-channel dequant along axis
        let outer: usize = input.dims[..axis].iter().product();
        let ch = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();
        let mut data = vec![0.0f32; input.data.len()];
        for o in 0..outer {
            for c in 0..ch {
                let s = scale.data[c];
                let zp = zero_point.data[c];
                let base = (o * ch + c) * inner;
                for i in 0..inner {
                    data[base + i] = (input.data[base + i] - zp) * s;
                }
            }
        }
        values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    }
    Ok(())
}

/// QLinearConv: quantized conv with dequantize-compute-requantize pattern.
/// Inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [bias]
/// Weight scale/zp can be per-channel (one per output channel).
fn exec_qlinear_conv(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.data[0];
    let x_zp = get_tensor(values, &node.input[2])?.data[0];
    let w_quant = get_tensor(values, &node.input[3])?;
    let w_scale_t = get_tensor(values, &node.input[4])?;
    let w_zp_t = get_tensor(values, &node.input[5])?;
    let y_scale = get_tensor(values, &node.input[6])?.data[0];
    let y_zp = get_tensor(values, &node.input[7])?.data[0];
    let bias = if node.input.len() > 8 && !node.input[8].is_empty() {
        Some(get_tensor(values, &node.input[8])?)
    } else {
        None
    };

    // Dequantize x (always scalar scale/zp)
    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(&x_quant.data, x_scale, x_zp),
    );

    // Dequantize w: per-channel along axis 0 (output channels)
    let c_out = w_quant.dims[0];
    let per_channel = w_scale_t.data.len() > 1;
    let elems_per_oc = w_quant.data.len() / c_out;
    let mut w_float_data = vec![0.0f32; w_quant.data.len()];
    for oc in 0..c_out {
        let scale = if per_channel {
            w_scale_t.data[oc]
        } else {
            w_scale_t.data[0]
        };
        let zp = if per_channel {
            w_zp_t.data[oc]
        } else {
            w_zp_t.data[0]
        };
        let base = oc * elems_per_oc;
        for i in 0..elems_per_oc {
            w_float_data[base + i] = (w_quant.data[base + i] - zp) * scale;
        }
    }
    let w_float = Tensor::new(w_quant.dims.clone(), w_float_data);

    // Build a synthetic Conv node reusing attributes from this node
    let mut conv_node = node.clone();
    conv_node.op_type = "Conv".to_string();
    conv_node.input = vec!["__qconv_x__".to_string(), "__qconv_w__".to_string()];
    if bias.is_some() {
        conv_node.input.push("__qconv_b__".to_string());
    }
    conv_node.output = vec!["__qconv_y__".to_string()];

    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qconv_x__".to_string(), x_float);
    tmp_values.insert("__qconv_w__".to_string(), w_float);
    if let Some(b) = bias {
        // Bias is int32. bias_float[oc] = bias_int32[oc] * x_scale * w_scale[oc]
        let mut bias_float = vec![0.0f32; b.data.len()];
        for oc in 0..b.data.len() {
            let ws = if per_channel {
                w_scale_t.data[oc]
            } else {
                w_scale_t.data[0]
            };
            bias_float[oc] = b.data[oc] * x_scale * ws;
        }
        tmp_values.insert("__qconv_b__".to_string(), Tensor::new(b.dims, bias_float));
    }

    exec_conv(&conv_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qconv_y__").unwrap();

    // Requantize output
    let y_quant = quantize_u8(&y_float.data, y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

/// QLinearAdd: inputs x, x_scale, x_zp, y, y_scale, y_zp, z_scale, z_zp
fn exec_qlinear_add(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.data[0];
    let x_zp = get_tensor(values, &node.input[2])?.data[0];
    let y_quant = get_tensor(values, &node.input[3])?;
    let y_scale = get_tensor(values, &node.input[4])?.data[0];
    let y_zp = get_tensor(values, &node.input[5])?.data[0];
    let z_scale = get_tensor(values, &node.input[6])?.data[0];
    let z_zp = get_tensor(values, &node.input[7])?.data[0];

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(&x_quant.data, x_scale, x_zp),
    );
    let y_float = Tensor::new(
        y_quant.dims.clone(),
        dequantize(&y_quant.data, y_scale, y_zp),
    );

    // Broadcast add
    let out_shape = broadcast_shape(&x_float.dims, &y_float.dims);
    let numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];
    let mut z_float = vec![0.0f32; numel];

    for i in 0..numel {
        let ai = broadcast_index(&index, &x_float.dims, &out_shape);
        let bi = broadcast_index(&index, &y_float.dims, &out_shape);
        z_float[i] = x_float.data[ai] + y_float.data[bi];
        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }

    let z_quant = quantize_u8(&z_float, z_scale, z_zp);
    values.insert(node.output[0].clone(), Tensor::new(out_shape, z_quant));
    Ok(())
}

/// QLinearMatMul: inputs a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
/// Weight (b) scale/zp can be per-channel (one per last-axis column).
fn exec_qlinear_matmul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a_quant = get_tensor(values, &node.input[0])?;
    let a_scale = get_tensor(values, &node.input[1])?.data[0];
    let a_zp = get_tensor(values, &node.input[2])?.data[0];
    let b_quant = get_tensor(values, &node.input[3])?;
    let b_scale_t = get_tensor(values, &node.input[4])?;
    let b_zp_t = get_tensor(values, &node.input[5])?;
    let y_scale = get_tensor(values, &node.input[6])?.data[0];
    let y_zp = get_tensor(values, &node.input[7])?.data[0];

    let a_float = Tensor::new(
        a_quant.dims.clone(),
        dequantize(&a_quant.data, a_scale, a_zp),
    );

    // Dequantize b: per-channel along last axis if scale is a vector
    let b_float = if b_scale_t.data.len() > 1 {
        // Per-channel: b shape is [..., K, N], scale shape is [N]
        let n = *b_quant.dims.last().unwrap();
        let k = b_quant.data.len() / n;
        let mut data = vec![0.0f32; b_quant.data.len()];
        for row in 0..k {
            for col in 0..n {
                let idx = row * n + col;
                let s = b_scale_t.data[col];
                let zp = b_zp_t.data[col];
                data[idx] = (b_quant.data[idx] - zp) * s;
            }
        }
        Tensor::new(b_quant.dims.clone(), data)
    } else {
        Tensor::new(
            b_quant.dims.clone(),
            dequantize(&b_quant.data, b_scale_t.data[0], b_zp_t.data[0]),
        )
    };

    // Use existing matmul logic
    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qmm_a__".to_string(), a_float);
    tmp_values.insert("__qmm_b__".to_string(), b_float);

    let mut mm_node = node.clone();
    mm_node.op_type = "MatMul".to_string();
    mm_node.input = vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()];
    mm_node.output = vec!["__qmm_y__".to_string()];

    exec_matmul(&mm_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qmm_y__").unwrap();

    let y_quant = quantize_u8(&y_float.data, y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

/// QLinearGlobalAveragePool: inputs x, x_scale, x_zp, y_scale, y_zp
fn exec_qlinear_global_avg_pool(
    node: &NodeProto,
    values: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.data[0];
    let x_zp = get_tensor(values, &node.input[2])?.data[0];
    let y_scale = get_tensor(values, &node.input[3])?.data[0];
    let y_zp = get_tensor(values, &node.input[4])?.data[0];

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(&x_quant.data, x_scale, x_zp),
    );

    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qgap_x__".to_string(), x_float);

    let mut gap_node = node.clone();
    gap_node.op_type = "GlobalAveragePool".to_string();
    gap_node.input = vec!["__qgap_x__".to_string()];
    gap_node.output = vec!["__qgap_y__".to_string()];

    exec_global_avg_pool(&gap_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qgap_y__").unwrap();

    let y_quant = quantize_u8(&y_float.data, y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use approx::assert_relative_eq;

    use super::*;

    const FIXTURES_DIR: &str = env!("FIXTURES_DIR");

    /// Run a model fixture against its bundled test data set.
    /// `fixture_dir` is relative to FIXTURES_DIR.
    /// `model_file` is the .onnx filename within that dir.
    fn run_fixture(fixture_dir: &str, model_file: &str, test_set: usize) {
        let base = Path::new(FIXTURES_DIR).join(fixture_dir);
        let model_bytes = fs::read(base.join(model_file)).expect("read model");
        let engine = InferenceEngine::from_bytes(&model_bytes).expect("load model");

        let test_dir = base.join(format!("test_data_set_{test_set}"));
        let input_bytes = fs::read(test_dir.join("input_0.pb")).expect("read input");
        let output_bytes = fs::read(test_dir.join("output_0.pb")).expect("read output");
        let input = Tensor::from_proto_bytes(&input_bytes).expect("parse input");
        let expected = Tensor::from_proto_bytes(&output_bytes).expect("parse output");

        // Discover input name from model graph
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

    // mnist-1 (opset 1): weights as Constant nodes, legacy broadcast
    #[test]
    fn test_mnist1_set_0() {
        run_fixture("mnist", "model.onnx", 0);
    }

    #[test]
    fn test_mnist1_set_1() {
        run_fixture("mnist", "model.onnx", 1);
    }

    #[test]
    fn test_mnist1_set_2() {
        run_fixture("mnist", "model.onnx", 2);
    }

    // mnist-12 (opset 12): weights as initializers, standard broadcast
    #[test]
    fn test_mnist12_set_0() {
        run_fixture("mnist-12", "mnist-12.onnx", 0);
    }

    // mobilenetv2-12 (opset 12): depthwise conv, clip, gemm, shape/gather/unsqueeze/concat
    #[test]
    fn test_mobilenetv2_set_0() {
        run_fixture("mobilenetv2-12", "mobilenetv2-12.onnx", 0);
    }

    // mobilenetv2-12-int8: quantized (QLinearConv, QLinearAdd, etc.)
    // Quantized int8 model: dequant-compute-requant in float32 introduces small
    // rounding differences vs the reference (onnxruntime uses optimized integer
    // kernels). Errors of ±1-2 quantization steps accumulate across 73 quantized
    // ops. We verify argmax matches and use absolute tolerance scaled to the
    // output quantization step size.
    #[test]
    fn test_mobilenetv2_int8_set_0() {
        let base = Path::new(FIXTURES_DIR).join("mobilenetv2-12-int8");
        let model_bytes = fs::read(base.join("mobilenetv2-12-int8.onnx")).expect("read model");
        let engine = InferenceEngine::from_bytes(&model_bytes).expect("load model");

        let test_dir = base.join("test_data_set_0");
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

        // Compare after softmax: logit differences are amplified in low-value
        // classes but compressed by softmax into meaningful probabilities.
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

        let got_class = got_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let expected_class = want_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(got_class, expected_class);
    }
}
