use std::collections::HashMap;
use std::ffi::c_void;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::layers::OpType;
use crate::onnx::NodeProto;
use crate::xnnpack_ffi::*;
use crate::{get_attr_float, get_attr_int, get_attr_ints, get_attr_string};

/// An ONNX op captured for compilation into an XNNPACK subgraph.
pub struct CapturedOp {
    pub op: OpType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub node: NodeProto,
}

/// A compiled XNNPACK subgraph that replaces a sequence of ONNX ops.
///
/// The subgraph handles NCHW↔NHWC layout conversion internally:
/// - 4D inputs are transposed from NCHW to NHWC at entry
/// - 4D outputs are transposed from NHWC to NCHW at exit
/// - Conv weights are transposed from OIHW to OHWI
pub struct XnnpackSubgraph {
    /// Compiled runtime (None until first execute for lazy compilation)
    compiled: Option<CompiledSubgraph>,
    /// Whether compilation was attempted and failed (use fallback)
    compile_failed: bool,
    /// Captured ops for deferred compilation
    ops: Vec<CapturedOp>,
    /// Required output names
    required_outputs: Vec<String>,
    /// Initializers needed for compilation
    initializers: HashMap<String, Tensor>,
    /// Pre-computed shape map (may be empty for lazy compilation)
    shape_hints: HashMap<String, Vec<usize>>,
    /// Fallback CPU plan nodes (used if XNNPACK compilation fails at runtime)
    pub fallback_nodes: Vec<super::PlanNode>,
}

struct CompiledSubgraph {
    runtime: xnn_runtime_t,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
    input_bufs: Vec<Vec<f32>>,
    output_bufs: Vec<Vec<f32>>,
    _static_data: Vec<Vec<f32>>,
}

// SAFETY: The xnn_runtime_t is not Sync but we only use it from one thread at a time.
// The Layer trait requires Send.
unsafe impl Send for XnnpackSubgraph {}
unsafe impl Send for CompiledSubgraph {}

impl Drop for CompiledSubgraph {
    fn drop(&mut self) {
        if !self.runtime.is_null() {
            unsafe { xnn_delete_runtime(self.runtime) };
        }
    }
}

/// Check if an OpType can be compiled into an XNNPACK subgraph node.
pub fn is_xnnpack_compatible(op: OpType) -> bool {
    matches!(
        op,
        OpType::Conv
            | OpType::Relu
            | OpType::Clip
            | OpType::Sigmoid
            | OpType::Tanh
            | OpType::Exp
            | OpType::Abs
            | OpType::Sqrt
            | OpType::Floor
            | OpType::Ceil
            | OpType::Log
            | OpType::LeakyRelu
            | OpType::Add
            | OpType::Sub
            | OpType::Mul
            | OpType::Div
            | OpType::Max
            | OpType::Min
            | OpType::MaxPool
            | OpType::GlobalAveragePool
            | OpType::Softmax
            | OpType::Gemm
            | OpType::MatMul
            | OpType::Flatten
            | OpType::Reshape
            | OpType::Concat
            | OpType::BatchNormalization
            | OpType::Neg
            | OpType::Sin
            | OpType::Cos
    )
}

/// Transpose conv weights from ONNX OIHW to XNNPACK OHWI format.
fn oihw_to_ohwi(data: &[f32], o: usize, i: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for oi in 0..o {
        for hi in 0..h {
            for wi in 0..w {
                for ii in 0..i {
                    out[((oi * h + hi) * w + wi) * i + ii] =
                        data[((oi * i + ii) * h + hi) * w + wi];
                }
            }
        }
    }
    out
}

/// Transpose depthwise conv weights from ONNX [C_out, 1, KH, KW] to
/// XNNPACK HWGo [1, KH, KW, C_out] format.
fn depthwise_oihw_to_hwgo(data: &[f32], c_out: usize, kh: usize, kw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for co in 0..c_out {
        for fh in 0..kh {
            for fw in 0..kw {
                out[(fh * kw + fw) * c_out + co] = data[(co * kh + fh) * kw + fw];
            }
        }
    }
    out
}

struct SubgraphBuilder {
    subgraph: xnn_subgraph_t,
    /// Maps tensor name → XNNPACK value ID
    value_ids: HashMap<String, u32>,
    _next_internal_id: u32,
    _num_external: u32,
    /// Holds static weight data that must outlive the subgraph/runtime
    static_data: Vec<Vec<f32>>,
    /// Track which values are in NHWC layout (vs NCHW or layout-agnostic)
    nhwc_values: std::collections::HashSet<String>,
}

impl SubgraphBuilder {
    fn new(num_external: u32) -> Result<Self> {
        ensure_init();
        let mut subgraph: xnn_subgraph_t = std::ptr::null_mut();
        let status = unsafe { xnn_create_subgraph(num_external, 0, &mut subgraph) };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_create_subgraph failed: {status:?}"
            )));
        }
        Ok(Self {
            subgraph,
            value_ids: HashMap::new(),
            _next_internal_id: num_external,
            _num_external: num_external,
            static_data: Vec::new(),
            nhwc_values: std::collections::HashSet::new(),
        })
    }

    fn define_external_input(
        &mut self,
        name: &str,
        external_id: u32,
        shape_nhwc: &[usize],
    ) -> Result<u32> {
        let mut id_out = 0u32;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype::xnn_datatype_fp32,
                shape_nhwc.len(),
                shape_nhwc.as_ptr(),
                std::ptr::null(),
                external_id,
                XNN_VALUE_FLAG_EXTERNAL_INPUT,
                &mut id_out,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_tensor_value (input {name}) failed: {status:?}"
            )));
        }
        self.value_ids.insert(name.to_string(), id_out);
        Ok(id_out)
    }

    fn define_external_output(
        &mut self,
        name: &str,
        external_id: u32,
        shape_nhwc: &[usize],
    ) -> Result<u32> {
        let mut id_out = 0u32;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype::xnn_datatype_fp32,
                shape_nhwc.len(),
                shape_nhwc.as_ptr(),
                std::ptr::null(),
                external_id,
                XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &mut id_out,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_tensor_value (output {name}) failed: {status:?}"
            )));
        }
        self.value_ids.insert(name.to_string(), id_out);
        Ok(id_out)
    }

    fn define_internal_value(&mut self, name: &str, shape: &[usize]) -> Result<u32> {
        let mut id_out = 0u32;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype::xnn_datatype_fp32,
                shape.len(),
                shape.as_ptr(),
                std::ptr::null(),
                XNN_INVALID_VALUE_ID,
                0,
                &mut id_out,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_tensor_value (internal {name}) failed: {status:?}"
            )));
        }
        self.value_ids.insert(name.to_string(), id_out);
        Ok(id_out)
    }

    fn define_static_value(&mut self, name: &str, shape: &[usize], data: Vec<f32>) -> Result<u32> {
        let mut id_out = 0u32;
        self.static_data.push(data);
        let data_ptr = self.static_data.last().unwrap().as_ptr() as *const c_void;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype::xnn_datatype_fp32,
                shape.len(),
                shape.as_ptr(),
                data_ptr,
                XNN_INVALID_VALUE_ID,
                0,
                &mut id_out,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_tensor_value (static {name}) failed: {status:?}"
            )));
        }
        self.value_ids.insert(name.to_string(), id_out);
        Ok(id_out)
    }

    fn get_or_define_value(
        &mut self,
        name: &str,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<u32> {
        if let Some(&id) = self.value_ids.get(name) {
            return Ok(id);
        }
        if let Some(shape) = shape_map.get(name) {
            self.define_internal_value(name, shape)
        } else {
            self.define_internal_value(name, &[])
        }
    }

    /// Ensure a value is in NHWC layout. If it's 4D and not already NHWC,
    /// insert a transpose and return the NHWC value ID.
    fn ensure_nhwc(&mut self, name: &str, shape_map: &HashMap<String, Vec<usize>>) -> Result<u32> {
        let src_id = self.get_or_define_value(name, shape_map)?;
        if self.nhwc_values.contains(name) {
            return Ok(src_id);
        }
        let shape = shape_map.get(name);
        if shape.is_none() || shape.unwrap().len() != 4 {
            return Ok(src_id);
        }
        let nchw_shape = shape.unwrap();
        let nhwc_name = format!("{name}__nhwc");
        if let Some(&id) = self.value_ids.get(&nhwc_name) {
            return Ok(id);
        }
        let nhwc_shape = [nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1]];
        let nhwc_id = self.define_internal_value(&nhwc_name, &nhwc_shape)?;
        let perm: [usize; 4] = [0, 2, 3, 1];
        let status = unsafe {
            xnn_define_static_transpose(self.subgraph, 4, perm.as_ptr(), src_id, nhwc_id, 0)
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_static_transpose (NCHW→NHWC for {name}) failed: {status:?}"
            )));
        }
        self.nhwc_values.insert(nhwc_name.clone());
        Ok(nhwc_id)
    }

    /// Mark a value as being in NHWC layout and define an NCHW version via transpose.
    fn define_nhwc_output(&mut self, name: &str, nhwc_shape: &[usize]) -> Result<u32> {
        let nhwc_name = format!("{name}__nhwc");
        let nhwc_id = self.define_internal_value(&nhwc_name, nhwc_shape)?;
        self.nhwc_values.insert(nhwc_name);

        // Define the NCHW output and add transpose
        let nchw_shape = [nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]];
        let nchw_id = self.get_or_define_value(name, &{
            let mut m = HashMap::new();
            m.insert(name.to_string(), nchw_shape.to_vec());
            m
        })?;
        let perm: [usize; 4] = [0, 3, 1, 2];
        let status = unsafe {
            xnn_define_static_transpose(self.subgraph, 4, perm.as_ptr(), nhwc_id, nchw_id, 0)
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_static_transpose (NHWC→NCHW for {name}) failed: {status:?}"
            )));
        }
        Ok(nhwc_id)
    }

    fn add_op(
        &mut self,
        captured: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        match captured.op {
            OpType::Conv => self.add_conv(captured, shape_map, initializers),
            OpType::Relu => self.add_relu(captured, shape_map),
            OpType::Clip => self.add_clip(captured, shape_map, initializers),
            OpType::Sigmoid => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_sigmoid,
                None,
            ),
            OpType::Tanh => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_tanh,
                None,
            ),
            OpType::Exp => {
                self.add_unary(captured, shape_map, xnn_unary_operator::xnn_unary_exp, None)
            }
            OpType::Abs => {
                self.add_unary(captured, shape_map, xnn_unary_operator::xnn_unary_abs, None)
            }
            OpType::Sqrt => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_square_root,
                None,
            ),
            OpType::Floor => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_floor,
                None,
            ),
            OpType::Ceil => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_ceiling,
                None,
            ),
            OpType::Log => {
                self.add_unary(captured, shape_map, xnn_unary_operator::xnn_unary_log, None)
            }
            OpType::Neg => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_negate,
                None,
            ),
            OpType::Sin => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_sine,
                None,
            ),
            OpType::Cos => self.add_unary(
                captured,
                shape_map,
                xnn_unary_operator::xnn_unary_cosine,
                None,
            ),
            OpType::LeakyRelu => self.add_leaky_relu(captured, shape_map),
            OpType::Add => {
                self.add_binary(captured, shape_map, xnn_binary_operator::xnn_binary_add)
            }
            OpType::Sub => self.add_binary(
                captured,
                shape_map,
                xnn_binary_operator::xnn_binary_subtract,
            ),
            OpType::Mul => self.add_binary(
                captured,
                shape_map,
                xnn_binary_operator::xnn_binary_multiply,
            ),
            OpType::Div => {
                self.add_binary(captured, shape_map, xnn_binary_operator::xnn_binary_divide)
            }
            OpType::Max => {
                self.add_binary(captured, shape_map, xnn_binary_operator::xnn_binary_maximum)
            }
            OpType::Min => {
                self.add_binary(captured, shape_map, xnn_binary_operator::xnn_binary_minimum)
            }
            OpType::MaxPool => self.add_maxpool(captured, shape_map),
            OpType::GlobalAveragePool => self.add_global_avg_pool(captured, shape_map),
            OpType::Softmax => self.add_softmax(captured, shape_map),
            OpType::Gemm => self.add_gemm(captured, shape_map, initializers),
            OpType::MatMul => self.add_matmul(captured, shape_map, initializers),
            OpType::Flatten => self.add_flatten(captured, shape_map),
            OpType::Reshape => self.add_reshape(captured, shape_map, initializers),
            OpType::Concat => self.add_concat(captured, shape_map),
            OpType::BatchNormalization => self.add_batch_norm(captured, shape_map, initializers),
            _ => Err(InferenceError::InvalidModel(format!(
                "XNNPACK: unsupported op {:?}",
                captured.op
            ))),
        }
    }

    fn add_conv(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let node = &cap.node;
        let input_name = &cap.inputs[0];
        let weight_name = &cap.inputs[1];
        let output_name = &cap.outputs[0];

        let weight = initializers.get(weight_name).ok_or_else(|| {
            InferenceError::InvalidModel(format!(
                "XNNPACK Conv: weight {weight_name} not found in initializers"
            ))
        })?;
        let w_shape = &weight.dims;
        let c_out = w_shape[0];
        let c_in_per_group = w_shape[1];
        let kh = w_shape[2];
        let kw = w_shape[3];

        let group = get_attr_int(node, "group").unwrap_or(1) as usize;
        let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let dilations = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
        let auto_pad = get_attr_string(node, "auto_pad").unwrap_or_default();

        let sh = strides[0] as u32;
        let sw = strides[1] as u32;
        let dh = dilations[0] as u32;
        let dw = dilations[1] as u32;

        let is_depthwise = group > 1 && c_in_per_group == 1;
        let c_out_per_group = c_out / group;

        // Compute SAME padding if needed
        let (pt, pl, pb, pr) = if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
            let in_shape = shape_map.get(input_name);
            if let Some(s) = in_shape {
                let h_in = s[2];
                let w_in = s[3];
                let oh = h_in.div_ceil(sh as usize);
                let ow = w_in.div_ceil(sw as usize);
                let pad_h =
                    ((oh - 1) * sh as usize + (dh as usize) * (kh - 1) + 1).saturating_sub(h_in);
                let pad_w =
                    ((ow - 1) * sw as usize + (dw as usize) * (kw - 1) + 1).saturating_sub(w_in);
                if auto_pad == "SAME_UPPER" {
                    (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
                } else {
                    (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2)
                }
            } else {
                (0, 0, 0, 0)
            }
        } else {
            (
                pads[0] as usize,
                pads[1] as usize,
                pads[2] as usize,
                pads[3] as usize,
            )
        };

        // Get NHWC input
        let input_id = self.ensure_nhwc(input_name, shape_map)?;

        // Transpose weights and define as static
        let weight_f = weight.floats();
        let filter_data = if is_depthwise {
            depthwise_oihw_to_hwgo(weight_f, c_out, kh, kw)
        } else {
            oihw_to_ohwi(weight_f, c_out, c_in_per_group, kh, kw)
        };
        let filter_shape = if is_depthwise {
            vec![1, kh, kw, c_out]
        } else {
            vec![c_out, kh, kw, c_in_per_group]
        };
        let filter_id =
            self.define_static_value(&format!("{weight_name}__xnn"), &filter_shape, filter_data)?;

        // Bias
        let bias_id = if cap.inputs.len() > 2 && !cap.inputs[2].is_empty() {
            let bias_name = &cap.inputs[2];
            if let Some(bias_tensor) = initializers.get(bias_name) {
                self.define_static_value(
                    &format!("{bias_name}__xnn"),
                    &[c_out],
                    bias_tensor.floats().to_vec(),
                )?
            } else {
                XNN_INVALID_VALUE_ID
            }
        } else {
            XNN_INVALID_VALUE_ID
        };

        // Compute output shape in NHWC
        let in_shape = shape_map.get(input_name).cloned().unwrap_or_default();
        let (h_out, w_out) = if in_shape.len() == 4 {
            let h_in = in_shape[2];
            let w_in = in_shape[3];
            let eff_kh = dh as usize * (kh - 1) + 1;
            let eff_kw = dw as usize * (kw - 1) + 1;
            (
                (h_in + pt + pb - eff_kh) / sh as usize + 1,
                (w_in + pl + pr - eff_kw) / sw as usize + 1,
            )
        } else {
            (1, 1)
        };
        let n = if in_shape.len() == 4 { in_shape[0] } else { 1 };
        let nhwc_out_shape = [n, h_out, w_out, c_out];
        let output_id = self.define_nhwc_output(output_name, &nhwc_out_shape)?;

        let status = if is_depthwise {
            let depth_multiplier = c_out_per_group;
            let input_channels = group;
            unsafe {
                xnn_define_depthwise_convolution_2d(
                    self.subgraph,
                    pt as u32,
                    pr as u32,
                    pb as u32,
                    pl as u32,
                    kh as u32,
                    kw as u32,
                    sh,
                    sw,
                    dh,
                    dw,
                    depth_multiplier as u32,
                    input_channels,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    input_id,
                    filter_id,
                    bias_id,
                    output_id,
                    0,
                )
            }
        } else {
            unsafe {
                xnn_define_convolution_2d(
                    self.subgraph,
                    pt as u32,
                    pr as u32,
                    pb as u32,
                    pl as u32,
                    kh as u32,
                    kw as u32,
                    sh,
                    sw,
                    dh,
                    dw,
                    group as u32,
                    c_in_per_group,
                    c_out_per_group,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    input_id,
                    filter_id,
                    bias_id,
                    output_id,
                    0,
                )
            }
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_convolution_2d failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_relu(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let params = xnn_unary_params {
            clamp: xnn_unary_clamp_params {
                min: 0.0,
                max: f32::INFINITY,
            },
        };
        self.add_unary(
            cap,
            shape_map,
            xnn_unary_operator::xnn_unary_clamp,
            Some(params),
        )
    }

    fn add_clip(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Opset ≥ 11: min/max come from tensor inputs[1] and inputs[2]
        // Opset < 11: min/max come from attributes
        let min_val = if cap.inputs.len() > 1 && !cap.inputs[1].is_empty() {
            if let Some(t) = initializers.get(&cap.inputs[1]) {
                if t.dtype() == crate::DType::Float {
                    t.floats()[0]
                } else {
                    f32::NEG_INFINITY
                }
            } else {
                f32::NEG_INFINITY
            }
        } else {
            get_attr_float(&cap.node, "min").unwrap_or(f32::NEG_INFINITY)
        };
        let max_val = if cap.inputs.len() > 2 && !cap.inputs[2].is_empty() {
            if let Some(t) = initializers.get(&cap.inputs[2]) {
                if t.dtype() == crate::DType::Float {
                    t.floats()[0]
                } else {
                    f32::INFINITY
                }
            } else {
                f32::INFINITY
            }
        } else {
            get_attr_float(&cap.node, "max").unwrap_or(f32::INFINITY)
        };
        let params = xnn_unary_params {
            clamp: xnn_unary_clamp_params {
                min: min_val,
                max: max_val,
            },
        };
        self.add_unary(
            cap,
            shape_map,
            xnn_unary_operator::xnn_unary_clamp,
            Some(params),
        )
    }

    fn add_leaky_relu(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let alpha = get_attr_float(&cap.node, "alpha").unwrap_or(0.01);
        let params = xnn_unary_params {
            leaky_relu: xnn_unary_leaky_relu_params {
                negative_slope: alpha,
            },
        };
        self.add_unary(
            cap,
            shape_map,
            xnn_unary_operator::xnn_unary_leaky_relu,
            Some(params),
        )
    }

    fn add_unary(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        op_type: xnn_unary_operator,
        params: Option<xnn_unary_params>,
    ) -> Result<()> {
        // If the input has an NHWC variant (from a preceding spatial op), operate
        // directly on that to avoid unnecessary NHWC→NCHW→NHWC transposes.
        let in_nhwc_name = format!("{}__nhwc", &cap.inputs[0]);
        let (input_id, is_nhwc) = if self.nhwc_values.contains(&cap.inputs[0]) {
            (self.get_or_define_value(&cap.inputs[0], shape_map)?, true)
        } else if let Some(&nhwc_id) = self.value_ids.get(&in_nhwc_name) {
            (nhwc_id, true)
        } else {
            (self.get_or_define_value(&cap.inputs[0], shape_map)?, false)
        };

        let output_id = if is_nhwc {
            // Use define_nhwc_output which creates the __nhwc value, defines
            // the NCHW version (or reuses existing external output), and adds
            // the NHWC→NCHW transpose.
            let out_name = &cap.outputs[0];
            let out_shape = shape_map.get(out_name).cloned().unwrap_or_default();
            if out_shape.len() == 4 {
                let nhwc_shape = [out_shape[0], out_shape[2], out_shape[3], out_shape[1]];
                self.define_nhwc_output(out_name, &nhwc_shape)?
            } else {
                self.get_or_define_value(&cap.outputs[0], shape_map)?
            }
        } else {
            self.get_or_define_value(&cap.outputs[0], shape_map)?
        };
        let params_ptr = params
            .as_ref()
            .map(|p| p as *const xnn_unary_params)
            .unwrap_or(std::ptr::null());
        let status =
            unsafe { xnn_define_unary(self.subgraph, op_type, params_ptr, input_id, output_id, 0) };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_unary failed for {:?}: {status:?}",
                cap.op
            )));
        }
        Ok(())
    }

    fn add_binary(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        op_type: xnn_binary_operator,
    ) -> Result<()> {
        let input1_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;

        // Handle legacy broadcasting (opset-1 broadcast=1 with axis attribute).
        // XNNPACK uses standard numpy broadcasting, so we need to reshape the second
        // input to align it at the right axis.
        let legacy_broadcast = get_attr_int(&cap.node, "broadcast").unwrap_or(0) != 0;
        let input2_id = if legacy_broadcast {
            let axis = get_attr_int(&cap.node, "axis").unwrap_or(0) as usize;
            let a_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
            let b_shape = shape_map.get(&cap.inputs[1]).cloned().unwrap_or_default();
            if b_shape.len() < a_shape.len() {
                // Reshape B from [D0..Dk] to [1..1, D0..Dk, 1..1] aligned at `axis`
                let mut new_shape = vec![1usize; a_shape.len()];
                for (i, &d) in b_shape.iter().enumerate() {
                    new_shape[axis + i] = d;
                }
                let reshaped_name = format!("{}__legacy_bc", &cap.inputs[1]);
                let reshaped_id = self.define_internal_value(&reshaped_name, &new_shape)?;
                let src_id = self.get_or_define_value(&cap.inputs[1], shape_map)?;
                let status = unsafe {
                    xnn_define_static_reshape(
                        self.subgraph,
                        new_shape.len(),
                        new_shape.as_ptr(),
                        src_id,
                        reshaped_id,
                        0,
                    )
                };
                if status != xnn_status::xnn_status_success {
                    return Err(InferenceError::InvalidModel(format!(
                        "xnn_define_static_reshape (legacy broadcast) failed: {status:?}"
                    )));
                }
                reshaped_id
            } else {
                self.get_or_define_value(&cap.inputs[1], shape_map)?
            }
        } else {
            self.get_or_define_value(&cap.inputs[1], shape_map)?
        };

        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;

        let status = unsafe {
            xnn_define_binary(
                self.subgraph,
                op_type,
                std::ptr::null(),
                input1_id,
                input2_id,
                output_id,
                0,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_binary failed for {:?}: {status:?}",
                cap.op
            )));
        }
        Ok(())
    }

    fn add_maxpool(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let node = &cap.node;
        let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
        let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

        let input_id = self.ensure_nhwc(&cap.inputs[0], shape_map)?;

        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let (n, c, h_in, w_in) = if in_shape.len() == 4 {
            (in_shape[0], in_shape[1], in_shape[2], in_shape[3])
        } else {
            (1, 1, 1, 1)
        };
        let kh = ks[0] as usize;
        let kw = ks[1] as usize;
        let sh = strides[0] as usize;
        let sw = strides[1] as usize;
        let h_out = (h_in + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
        let w_out = (w_in + pads[1] as usize + pads[3] as usize - kw) / sw + 1;

        let nhwc_out_shape = [n, h_out, w_out, c];
        let output_id = self.define_nhwc_output(&cap.outputs[0], &nhwc_out_shape)?;

        let status = unsafe {
            xnn_define_max_pooling_2d(
                self.subgraph,
                pads[0] as u32,
                pads[3] as u32,
                pads[2] as u32,
                pads[1] as u32,
                kh as u32,
                kw as u32,
                sh as u32,
                sw as u32,
                1,
                1, // dilation
                f32::NEG_INFINITY,
                f32::INFINITY,
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_max_pooling_2d failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_global_avg_pool(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let input_id = self.ensure_nhwc(&cap.inputs[0], shape_map)?;

        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let (n, c) = if in_shape.len() == 4 {
            (in_shape[0], in_shape[1])
        } else {
            (1, 1)
        };
        let nhwc_out_shape = [n, 1, 1, c];
        let output_id = self.define_nhwc_output(&cap.outputs[0], &nhwc_out_shape)?;

        let status = unsafe {
            xnn_define_global_average_pooling_2d(
                self.subgraph,
                f32::NEG_INFINITY,
                f32::INFINITY,
                input_id,
                output_id,
                XNN_FLAG_KEEP_DIMS,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_global_average_pooling_2d failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_softmax(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe { xnn_define_softmax(self.subgraph, input_id, output_id, 0) };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_softmax failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_gemm(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let node = &cap.node;
        let trans_b = get_attr_int(node, "transB").unwrap_or(0) != 0;

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;

        // Weight
        let weight_name = &cap.inputs[1];
        let weight = initializers.get(weight_name).ok_or_else(|| {
            InferenceError::InvalidModel(format!(
                "XNNPACK Gemm: weight {weight_name} not in initializers"
            ))
        })?;
        let w_data = weight.floats().to_vec();
        let w_shape: Vec<usize> = weight.dims.iter().copied().collect();
        let filter_id =
            self.define_static_value(&format!("{weight_name}__xnn"), &w_shape, w_data)?;

        // Bias
        let bias_id = if cap.inputs.len() > 2 && !cap.inputs[2].is_empty() {
            let bias_name = &cap.inputs[2];
            if let Some(bias_tensor) = initializers.get(bias_name) {
                self.define_static_value(
                    &format!("{bias_name}__xnn"),
                    &bias_tensor.dims.iter().copied().collect::<Vec<_>>(),
                    bias_tensor.floats().to_vec(),
                )?
            } else {
                XNN_INVALID_VALUE_ID
            }
        } else {
            XNN_INVALID_VALUE_ID
        };

        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;

        let flags = if trans_b {
            XNN_FLAG_TRANSPOSE_WEIGHTS
        } else {
            0
        };
        let status = unsafe {
            xnn_define_fully_connected(
                self.subgraph,
                f32::NEG_INFINITY,
                f32::INFINITY,
                input_id,
                filter_id,
                bias_id,
                output_id,
                flags,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_fully_connected (Gemm) failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_matmul(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let a_name = &cap.inputs[0];
        let b_name = &cap.inputs[1];

        // If B is a static weight and both are 2D, use fully_connected
        let b_shape = shape_map.get(b_name).cloned().unwrap_or_default();
        let a_shape = shape_map.get(a_name).cloned().unwrap_or_default();
        if a_shape.len() == 2 && b_shape.len() == 2 && initializers.contains_key(b_name) {
            let input_id = self.get_or_define_value(a_name, shape_map)?;
            let weight = initializers.get(b_name).unwrap();
            // MatMul: A[M,K] @ B[K,N] = C[M,N]
            // fully_connected expects filter [N,K] (output_channels, input_channels)
            // B is [K,N], so we pass it transposed
            let filter_id = self.define_static_value(
                &format!("{b_name}__xnn"),
                &b_shape,
                weight.floats().to_vec(),
            )?;
            let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
            let status = unsafe {
                xnn_define_fully_connected(
                    self.subgraph,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    input_id,
                    filter_id,
                    XNN_INVALID_VALUE_ID,
                    output_id,
                    XNN_FLAG_TRANSPOSE_WEIGHTS,
                )
            };
            if status != xnn_status::xnn_status_success {
                return Err(InferenceError::InvalidModel(format!(
                    "xnn_define_fully_connected (MatMul) failed: {status:?}"
                )));
            }
        } else {
            // Use batch_matrix_multiply for higher-rank or non-static MatMul
            let input1_id = self.get_or_define_value(a_name, shape_map)?;
            let input2_id = self.get_or_define_value(b_name, shape_map)?;
            let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
            let status = unsafe {
                xnn_define_batch_matrix_multiply(self.subgraph, input1_id, input2_id, output_id, 0)
            };
            if status != xnn_status::xnn_status_success {
                return Err(InferenceError::InvalidModel(format!(
                    "xnn_define_batch_matrix_multiply failed: {status:?}"
                )));
            }
        }
        Ok(())
    }

    fn add_flatten(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let axis = get_attr_int(&cap.node, "axis").unwrap_or(1) as usize;
        let outer: usize = in_shape[..axis].iter().product();
        let inner: usize = in_shape[axis..].iter().product();
        let new_shape = [outer, inner];

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_static_reshape(self.subgraph, 2, new_shape.as_ptr(), input_id, output_id, 0)
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_static_reshape (Flatten) failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_reshape(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        _initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let out_shape = shape_map.get(&cap.outputs[0]).cloned().unwrap_or_default();
        if out_shape.is_empty() || out_shape.len() > XNN_MAX_TENSOR_DIMS {
            return Err(InferenceError::InvalidModel(
                "XNNPACK Reshape: unknown or too many output dims".into(),
            ));
        }

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_static_reshape(
                self.subgraph,
                out_shape.len(),
                out_shape.as_ptr(),
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_static_reshape failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_concat(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let raw_axis = get_attr_int(&cap.node, "axis").unwrap_or(0);
        // Resolve negative axis using output rank
        let out_shape = shape_map.get(&cap.outputs[0]).cloned().unwrap_or_default();
        let ndim = out_shape.len() as i64;
        let axis = if raw_axis < 0 {
            raw_axis + ndim
        } else {
            raw_axis
        } as usize;

        let mut input_ids = Vec::new();
        for name in &cap.inputs {
            input_ids.push(self.get_or_define_value(name, shape_map)?);
        }
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_concatenate(
                self.subgraph,
                axis,
                input_ids.len(),
                input_ids.as_ptr(),
                output_id,
                0,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_concatenate failed: {status:?}"
            )));
        }
        Ok(())
    }

    fn add_batch_norm(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Fuse BatchNorm into scale+bias: y = (x - mean) / sqrt(var + eps) * gamma + beta
        // Which is: y = x * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var + eps))
        let epsilon = get_attr_float(&cap.node, "epsilon").unwrap_or(1e-5);

        let scale_t = initializers.get(&cap.inputs[1]).ok_or_else(|| {
            InferenceError::InvalidModel("XNNPACK BatchNorm: scale not in initializers".into())
        })?;
        let bias_t = initializers.get(&cap.inputs[2]).ok_or_else(|| {
            InferenceError::InvalidModel("XNNPACK BatchNorm: bias not in initializers".into())
        })?;
        let mean_t = initializers.get(&cap.inputs[3]).ok_or_else(|| {
            InferenceError::InvalidModel("XNNPACK BatchNorm: mean not in initializers".into())
        })?;
        let var_t = initializers.get(&cap.inputs[4]).ok_or_else(|| {
            InferenceError::InvalidModel("XNNPACK BatchNorm: var not in initializers".into())
        })?;

        let gamma = scale_t.floats();
        let beta = bias_t.floats();
        let mean = mean_t.floats();
        let var = var_t.floats();
        let c = gamma.len();

        // Compute fused scale and bias
        let mut fused_scale = vec![0.0f32; c];
        let mut fused_bias = vec![0.0f32; c];
        for i in 0..c {
            let inv_std = 1.0 / (var[i] + epsilon).sqrt();
            fused_scale[i] = gamma[i] * inv_std;
            fused_bias[i] = beta[i] - mean[i] * gamma[i] * inv_std;
        }

        // Implement as: multiply by scale (broadcast), then add bias (broadcast)
        // Reshape scale/bias to [1,C,1,1] for correct NCHW broadcasting in XNNPACK
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let param_shape = if in_shape.len() == 4 {
            vec![1, c, 1, 1]
        } else {
            vec![c]
        };
        let scale_id = self.define_static_value(
            &format!("{}__bn_scale", &cap.outputs[0]),
            &param_shape,
            fused_scale,
        )?;
        let bias_data_id = self.define_static_value(
            &format!("{}__bn_bias", &cap.outputs[0]),
            &param_shape,
            fused_bias,
        )?;

        // Intermediate: x * scale
        let mid_name = format!("{}__bn_mid", &cap.outputs[0]);
        let mid_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let mid_id = self.define_internal_value(&mid_name, &mid_shape)?;

        let status = unsafe {
            xnn_define_binary(
                self.subgraph,
                xnn_binary_operator::xnn_binary_multiply,
                std::ptr::null(),
                input_id,
                scale_id,
                mid_id,
                0,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_binary (BatchNorm mul) failed: {status:?}"
            )));
        }

        // Output: mid + bias
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_binary(
                self.subgraph,
                xnn_binary_operator::xnn_binary_add,
                std::ptr::null(),
                mid_id,
                bias_data_id,
                output_id,
                0,
            )
        };
        if status != xnn_status::xnn_status_success {
            return Err(InferenceError::InvalidModel(format!(
                "xnn_define_binary (BatchNorm add) failed: {status:?}"
            )));
        }

        // Propagate NHWC status
        if self.nhwc_values.contains(&cap.inputs[0]) {
            self.nhwc_values.insert(mid_name);
            self.nhwc_values.insert(cap.outputs[0].clone());
        }

        Ok(())
    }
}

impl XnnpackSubgraph {
    /// Create a new lazy XNNPACK subgraph that compiles on first execution.
    pub fn new(
        ops: Vec<CapturedOp>,
        required_outputs: Vec<String>,
        shape_hints: HashMap<String, Vec<usize>>,
        initializers: HashMap<String, Tensor>,
    ) -> Self {
        Self {
            compiled: None,
            compile_failed: false,
            ops,
            required_outputs,
            initializers,
            shape_hints,
            fallback_nodes: Vec::new(),
        }
    }
}

/// Build a compiled XNNPACK subgraph from a sequence of captured ops.
///
/// All external values use NCHW format (our engine's native layout).
/// Layout conversion (NCHW↔NHWC) is handled internally by XNNPACK transpose nodes.
fn compile_subgraph(
    ops: &[CapturedOp],
    required_outputs: &[String],
    shape_map: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, Tensor>,
) -> Result<CompiledSubgraph> {
    // Identify external inputs
    let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();

    let init_names: std::collections::HashSet<&str> =
        initializers.keys().map(|s| s.as_str()).collect();

    for op in ops {
        for out in &op.outputs {
            produced.insert(out.clone());
        }
    }
    for op in ops {
        for inp in &op.inputs {
            if !inp.is_empty() && !produced.contains(inp) && !init_names.contains(inp.as_str()) {
                consumed.insert(inp.clone());
            }
        }
    }

    let external_inputs: Vec<String> = consumed.into_iter().collect();

    tracing::debug!(
        "XNNPACK compile_subgraph: external_inputs={:?}",
        external_inputs
            .iter()
            .map(|n| (n, shape_map.get(n)))
            .collect::<Vec<_>>()
    );
    tracing::debug!(
        "XNNPACK compile_subgraph: required_outputs={:?}",
        required_outputs
            .iter()
            .map(|n| (n, shape_map.get(n)))
            .collect::<Vec<_>>()
    );

    let num_external = (external_inputs.len() + required_outputs.len()) as u32;
    let mut builder = SubgraphBuilder::new(num_external)?;

    // Define external inputs in NCHW shape (our native format).
    // Internal transpose nodes added by ensure_nhwc will convert to NHWC for spatial ops.
    let mut input_shapes = Vec::new();
    let mut input_bufs = Vec::new();
    for (i, name) in external_inputs.iter().enumerate() {
        let shape = shape_map.get(name).cloned().unwrap_or_default();
        let numel: usize = shape.iter().product();
        builder.define_external_input(name, i as u32, &shape)?;
        input_bufs.push(vec![0.0f32; numel + XNN_EXTRA_BYTES / 4]);
        input_shapes.push(shape);
    }

    // Define external outputs in NCHW shape.
    // Internal transpose nodes added by define_nhwc_output convert NHWC→NCHW.
    let mut output_shapes = Vec::new();
    let mut output_bufs = Vec::new();
    for (i, name) in required_outputs.iter().enumerate() {
        let shape = shape_map.get(name).cloned().unwrap_or_default();
        let numel: usize = shape.iter().product();
        let ext_id = (external_inputs.len() + i) as u32;
        builder.define_external_output(name, ext_id, &shape)?;
        output_bufs.push(vec![0.0f32; numel + XNN_EXTRA_BYTES / 4]);
        output_shapes.push(shape);
    }

    // Pre-define initializers consumed by ops as static values.
    // Conv/Gemm/BatchNorm handle their own weights explicitly, but other ops
    // (e.g., Add with a bias initializer) need them defined upfront.
    for op in ops {
        for inp in &op.inputs {
            if !inp.is_empty() && !builder.value_ids.contains_key(inp) && !produced.contains(inp) {
                if let Some(tensor) = initializers.get(inp) {
                    if tensor.dtype() == crate::DType::Float {
                        let shape: Vec<usize> = tensor.dims.iter().copied().collect();
                        let data = tensor.floats().to_vec();
                        builder.define_static_value(inp, &shape, data)?;
                    }
                }
            }
        }
    }

    // Add all ops
    for op in ops {
        builder.add_op(op, shape_map, initializers)?;
    }

    // Create runtime
    let mut runtime: xnn_runtime_t = std::ptr::null_mut();
    let status = unsafe { xnn_create_runtime(builder.subgraph, &mut runtime) };
    unsafe { xnn_delete_subgraph(builder.subgraph) };

    if status != xnn_status::xnn_status_success {
        return Err(InferenceError::InvalidModel(format!(
            "xnn_create_runtime failed: {status:?}"
        )));
    }

    Ok(CompiledSubgraph {
        runtime,
        input_names: external_inputs,
        output_names: required_outputs.to_vec(),
        input_shapes,
        output_shapes,
        input_bufs,
        output_bufs,
        _static_data: builder.static_data,
    })
}

impl XnnpackSubgraph {
    fn ensure_compiled(&mut self, values: &HashMap<String, Tensor>) -> Result<()> {
        if self.compiled.is_some() || self.compile_failed {
            return Ok(());
        }

        // Build shape_map from actual runtime values + shape_hints + initializers
        let mut shape_map = self.shape_hints.clone();
        for (name, tensor) in &self.initializers {
            if !shape_map.contains_key(name) && !tensor.dims.is_empty() {
                shape_map.insert(name.clone(), tensor.dims.to_vec());
            }
        }
        for op in &self.ops {
            for name in op.inputs.iter().chain(op.outputs.iter()) {
                if !name.is_empty() && !shape_map.contains_key(name) {
                    if let Some(t) = values.get(name) {
                        if !t.dims.is_empty() {
                            shape_map.insert(name.clone(), t.dims.to_vec());
                        }
                    }
                }
            }
        }

        // Check for non-float external inputs — XNNPACK only supports f32.
        // Identify which inputs are external (not produced by ops, not initializers).
        {
            let mut produced: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for op in &self.ops {
                for out in &op.outputs {
                    produced.insert(out);
                }
            }
            for op in &self.ops {
                for inp in &op.inputs {
                    if !inp.is_empty()
                        && !produced.contains(inp.as_str())
                        && !self.initializers.contains_key(inp)
                    {
                        if let Some(t) = values.get(inp) {
                            if t.dtype() != crate::DType::Float {
                                self.compile_failed = true;
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }

        // Infer output shapes from ops using input shapes
        for op in &self.ops {
            let inferred = op.op.infer_output_shape(
                &op.node,
                &op.inputs,
                &shape_map
                    .iter()
                    .map(|(k, v)| (k.clone(), crate::Dims::from(v.as_slice())))
                    .collect(),
                &self.initializers,
            );
            if let Some(dims) = inferred {
                for out in &op.outputs {
                    if !out.is_empty() && !shape_map.contains_key(out) {
                        shape_map.insert(out.clone(), dims.to_vec());
                    }
                }
            }
        }

        // Verify all input and output shapes are known and non-zero
        let mut missing_shape = false;
        for op in &self.ops {
            for name in op.inputs.iter().chain(op.outputs.iter()) {
                if name.is_empty() || self.initializers.contains_key(name) {
                    continue;
                }
                match shape_map.get(name) {
                    Some(shape) if !shape.is_empty() && shape.iter().all(|&d| d > 0) => {}
                    other => {
                        tracing::debug!("XNNPACK: missing/zero shape for '{name}': {:?}", other);
                        missing_shape = true;
                    }
                }
            }
        }
        if missing_shape {
            tracing::debug!(
                "XNNPACK: fallback due to missing shapes, ops: {:?}",
                self.ops
                    .iter()
                    .map(|o| format!("{:?}", o.op))
                    .collect::<Vec<_>>()
            );
            self.compile_failed = true;
            return Ok(());
        }

        match compile_subgraph(
            &self.ops,
            &self.required_outputs,
            &shape_map,
            &self.initializers,
        ) {
            Ok(compiled) => {
                tracing::debug!(
                    "XNNPACK: compiled subgraph: {} inputs {:?}, {} outputs {:?}, ops: {:?}",
                    compiled.input_names.len(),
                    compiled.input_shapes,
                    compiled.output_names.len(),
                    compiled.output_shapes,
                    self.ops
                        .iter()
                        .map(|o| format!("{:?}", o.op))
                        .collect::<Vec<_>>()
                );
                self.compiled = Some(compiled);
                Ok(())
            }
            Err(_e) => {
                tracing::debug!("XNNPACK: compile failed: {_e}");
                self.compile_failed = true;
                Ok(())
            }
        }
    }

    pub fn execute(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let _span = tracing::trace_span!("xnnpack_subgraph").entered();

        self.ensure_compiled(values)?;

        if self.compile_failed {
            return self.execute_fallback(values);
        }

        let compiled = self.compiled.as_ref().unwrap();

        // Verify runtime input shapes match compiled shapes; fall back if mismatch
        for (i, name) in compiled.input_names.iter().enumerate() {
            if let Some(tensor) = values.get(name) {
                if tensor.dtype() != crate::DType::Float {
                    tracing::debug!(
                        "XNNPACK: dtype mismatch for input '{name}': {:?}",
                        tensor.dtype()
                    );
                    self.compiled = None;
                    self.compile_failed = true;
                    return self.execute_fallback(values);
                }
                let runtime_shape: Vec<usize> = tensor.dims.to_vec();
                if runtime_shape != compiled.input_shapes[i] {
                    tracing::debug!(
                        "XNNPACK: shape mismatch for input '{name}': runtime {:?} != compiled {:?}",
                        runtime_shape,
                        compiled.input_shapes[i]
                    );
                    self.compiled = None;
                    self.compile_failed = true;
                    return self.execute_fallback(values);
                }
            }
        }

        let compiled = self.compiled.as_mut().unwrap();

        let num_external = compiled.input_names.len() + compiled.output_names.len();
        let mut external_values = Vec::with_capacity(num_external);

        // Copy input data into buffers (with XNN_EXTRA_BYTES padding)
        for (i, name) in compiled.input_names.iter().enumerate() {
            let tensor = values.get(name).ok_or_else(|| {
                InferenceError::InvalidModel(format!("XNNPACK: missing input {name}"))
            })?;
            let src = tensor.floats();
            let buf = &mut compiled.input_bufs[i];
            let needed = src.len() + XNN_EXTRA_BYTES / 4;
            if buf.len() < needed {
                buf.resize(needed, 0.0);
            }
            buf[..src.len()].copy_from_slice(src);

            external_values.push(xnn_external_value {
                id: i as u32,
                data: buf.as_mut_ptr() as *mut c_void,
            });
        }

        // Prepare output buffers
        for (i, _name) in compiled.output_names.iter().enumerate() {
            let ext_id = (compiled.input_names.len() + i) as u32;
            let buf = &mut compiled.output_bufs[i];
            external_values.push(xnn_external_value {
                id: ext_id,
                data: buf.as_mut_ptr() as *mut c_void,
            });
        }

        let status = unsafe { xnn_reshape_runtime(compiled.runtime) };
        if status != xnn_status::xnn_status_success {
            tracing::debug!("XNNPACK: reshape failed with {:?}", status);
            self.compiled = None;
            self.compile_failed = true;
            return self.execute_fallback(values);
        }

        let status = unsafe {
            xnn_setup_runtime_v2(
                compiled.runtime,
                external_values.len(),
                external_values.as_ptr(),
            )
        };
        if status != xnn_status::xnn_status_success {
            tracing::debug!("XNNPACK: setup failed with {:?}", status);
            self.compiled = None;
            self.compile_failed = true;
            return self.execute_fallback(values);
        }

        let status = unsafe { xnn_invoke_runtime(compiled.runtime) };
        if status != xnn_status::xnn_status_success {
            tracing::debug!("XNNPACK: invoke failed with {:?}", status);
            self.compiled = None;
            self.compile_failed = true;
            return self.execute_fallback(values);
        }

        // Copy outputs back
        for (i, name) in compiled.output_names.iter().enumerate() {
            let shape = &compiled.output_shapes[i];
            let numel: usize = shape.iter().product();
            let buf = &compiled.output_bufs[i];

            let out_tensor = values.entry(name.clone()).or_default();
            let out_data = out_tensor.as_mut_f32(numel);
            out_data.copy_from_slice(&buf[..numel]);
            out_tensor.set_dims(shape);
        }

        Ok(())
    }

    fn execute_fallback(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        tracing::debug!(
            "XNNPACK: falling back to CPU for {} ops",
            self.fallback_nodes.len()
        );
        for node in &mut self.fallback_nodes {
            match node {
                super::PlanNode::Single { output, layer } => {
                    if output.is_empty() {
                        continue;
                    }
                    let (key, mut out) = values
                        .remove_entry(output.as_str())
                        .unwrap_or_else(|| (output.clone(), Tensor::default()));
                    let result = layer.execute(values, &mut out);
                    values.insert(key, out);
                    result?;
                }
                super::PlanNode::Loop(l) => l.execute(values)?,
                super::PlanNode::Split(s) => s.execute(values)?,
                super::PlanNode::If(i) => i.execute(values)?,
                super::PlanNode::TopK(t) => t.execute(values)?,
                super::PlanNode::Scan(s) => s.execute(values)?,
                #[cfg(feature = "xnnpack")]
                super::PlanNode::XnnpackSubgraph(sg) => sg.execute(values)?,
            }
        }
        Ok(())
    }

    pub fn output_names(&self) -> &[String] {
        &self.required_outputs
    }
}
