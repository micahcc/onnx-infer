//! XNNPACK subgraph compilation.
//!
//! Compiles a sequence of ONNX ops into an XNNPACK subgraph runtime.
//! Assumes the graph has been through `graph_opt::optimize()`, which:
//! - Inserts NCHW→NHWC transposes before spatial ops
//! - Inserts NHWC→NCHW transposes after spatial ops
//! - Folds BatchNorm into Conv
//! - Eliminates redundant transpose pairs
//!
//! This module does NOT track NHWC layout per tensor — graph-opt handles it.
//! Transpose nodes from graph-opt are compiled as `xnn_define_static_transpose`.
//! Conv/MaxPool/etc receive NHWC input directly.

use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::c_void;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::layers::OpType;
use crate::onnx_ir::Node;
use crate::xnnpack_ffi::*;

// ---------------------------------------------------------------------------
// Helper structs
// ---------------------------------------------------------------------------

struct Padding2D {
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
}

struct Stride2D {
    h: u32,
    w: u32,
}

struct Dilation2D {
    h: u32,
    w: u32,
}

// ---------------------------------------------------------------------------
// Weight transpose helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An ONNX op captured for compilation into an XNNPACK subgraph.
pub struct CapturedOp {
    pub op: OpType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub node: Node,
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
            | OpType::Neg
            | OpType::Sin
            | OpType::Cos
            | OpType::Add
            | OpType::Sub
            | OpType::Mul
            | OpType::Div
            | OpType::Max
            | OpType::Min
            | OpType::MaxPool
            | OpType::AveragePool
            | OpType::GlobalAveragePool
            | OpType::Softmax
            | OpType::Gemm
            | OpType::MatMul
            | OpType::Flatten
            | OpType::Reshape
            | OpType::Concat
            | OpType::Transpose
            | OpType::LayoutTranspose
            | OpType::Identity
            | OpType::Unsqueeze
            | OpType::Squeeze
            | OpType::Slice
            | OpType::Resize
            | OpType::Round
            | OpType::Cast
            | OpType::ReduceMin
    )
}

// ---------------------------------------------------------------------------
// SubgraphBuilder — creates XNNPACK subgraph from captured ops
// ---------------------------------------------------------------------------

struct SubgraphBuilder {
    subgraph: xnn_subgraph_t,
    value_ids: HashMap<String, u32>,
    static_data: Vec<Vec<f32>>,
    static_data_bytes: Vec<Vec<u8>>,
}

impl SubgraphBuilder {
    fn new(num_external: u32) -> Result<Self> {
        ensure_init();
        let mut subgraph: xnn_subgraph_t = std::ptr::null_mut();
        let status = unsafe { xnn_create_subgraph(num_external, 0, &mut subgraph) };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_create_subgraph failed: {status:?}");
        }
        Ok(Self {
            subgraph,
            value_ids: HashMap::new(),
            static_data: Vec::new(),
            static_data_bytes: Vec::new(),
        })
    }

    fn define_external_input(
        &mut self,
        name: &str,
        external_id: u32,
        shape: &[usize],
    ) -> Result<u32> {
        let mut id_out = 0u32;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype_xnn_datatype_fp32,
                shape.len(),
                shape.as_ptr(),
                std::ptr::null(),
                external_id,
                XNN_VALUE_FLAG_EXTERNAL_INPUT,
                &mut id_out,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_tensor_value (input {name}) failed: {status:?}");
        }
        self.value_ids.insert(name.to_string(), id_out);
        Ok(id_out)
    }

    fn define_external_output(
        &mut self,
        name: &str,
        external_id: u32,
        shape: &[usize],
    ) -> Result<u32> {
        let mut id_out = 0u32;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype_xnn_datatype_fp32,
                shape.len(),
                shape.as_ptr(),
                std::ptr::null(),
                external_id,
                XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &mut id_out,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_tensor_value (output {name}) failed: {status:?}");
        }
        self.value_ids.insert(name.to_string(), id_out);
        Ok(id_out)
    }

    fn define_internal_value(&mut self, name: &str, shape: &[usize]) -> Result<u32> {
        let mut id_out = 0u32;
        let status = unsafe {
            xnn_define_tensor_value(
                self.subgraph,
                xnn_datatype_xnn_datatype_fp32,
                shape.len(),
                shape.as_ptr(),
                std::ptr::null(),
                XNN_INVALID_VALUE_ID,
                0,
                &mut id_out,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_tensor_value (internal {name}) failed: {status:?}");
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
                xnn_datatype_xnn_datatype_fp32,
                shape.len(),
                shape.as_ptr(),
                data_ptr,
                XNN_INVALID_VALUE_ID,
                0,
                &mut id_out,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_tensor_value (static {name}) failed: {status:?}");
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
        let shape = shape_map.get(name).cloned().unwrap_or_default();
        self.define_internal_value(name, &shape)
    }

    // -----------------------------------------------------------------------
    // Op dispatch
    // -----------------------------------------------------------------------

    fn add_op(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        match cap.op {
            OpType::Conv => self.add_conv(cap, shape_map, initializers),
            OpType::Relu => self.add_relu(cap, shape_map),
            OpType::Clip => self.add_clip(cap, shape_map, initializers),
            OpType::Sigmoid => {
                self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_sigmoid, None)
            }
            OpType::Tanh => self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_tanh, None),
            OpType::Exp => self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_exp, None),
            OpType::Abs => self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_abs, None),
            OpType::Sqrt => self.add_unary(
                cap,
                shape_map,
                xnn_unary_operator_xnn_unary_square_root,
                None,
            ),
            OpType::Floor => {
                self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_floor, None)
            }
            OpType::Ceil => {
                self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_ceiling, None)
            }
            OpType::Log => self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_log, None),
            OpType::Neg => {
                self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_negate, None)
            }
            OpType::Sin => self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_sine, None),
            OpType::Cos => {
                self.add_unary(cap, shape_map, xnn_unary_operator_xnn_unary_cosine, None)
            }
            OpType::LeakyRelu => self.add_leaky_relu(cap, shape_map),
            OpType::Add => self.add_binary(cap, shape_map, xnn_binary_operator_xnn_binary_add),
            OpType::Sub => self.add_binary(cap, shape_map, xnn_binary_operator_xnn_binary_subtract),
            OpType::Mul => self.add_binary(cap, shape_map, xnn_binary_operator_xnn_binary_multiply),
            OpType::Div => self.add_binary(cap, shape_map, xnn_binary_operator_xnn_binary_divide),
            OpType::Max => self.add_binary(cap, shape_map, xnn_binary_operator_xnn_binary_maximum),
            OpType::Min => self.add_binary(cap, shape_map, xnn_binary_operator_xnn_binary_minimum),
            OpType::MaxPool => self.add_maxpool(cap, shape_map),
            OpType::AveragePool => self.add_average_pool(cap, shape_map),
            OpType::GlobalAveragePool => self.add_global_avg_pool(cap, shape_map),
            OpType::Softmax => self.add_softmax(cap, shape_map),
            OpType::Gemm => self.add_gemm(cap, shape_map, initializers),
            OpType::MatMul => self.add_matmul(cap, shape_map, initializers),
            OpType::Flatten => self.add_flatten(cap, shape_map),
            OpType::Reshape => self.add_reshape(cap, shape_map),
            OpType::Concat => self.add_concat(cap, shape_map),
            OpType::Transpose | OpType::LayoutTranspose => self.add_transpose(cap, shape_map),
            OpType::Identity => self.add_identity(cap, shape_map),
            OpType::Unsqueeze => self.add_static_reshape(cap, shape_map),
            OpType::Squeeze => self.add_static_reshape(cap, shape_map),
            OpType::Slice => self.add_slice(cap, shape_map),
            OpType::Resize => self.add_resize(cap, shape_map),
            OpType::Round => self.add_round(cap, shape_map),
            OpType::Cast => self.add_cast(cap, shape_map),
            OpType::ReduceMin => self.add_reduce_min(cap, shape_map),
            _ => anyhow::bail!("XNNPACK: unsupported op {:?}", cap.op),
        }
    }

    // -----------------------------------------------------------------------
    // Spatial ops — receive NHWC input (graph-opt inserted transposes)
    // -----------------------------------------------------------------------

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
            anyhow::anyhow!("XNNPACK Conv: weight {weight_name} not found in initializers")
        })?;
        let w_shape = &weight.dims;
        let c_out = w_shape[0];
        let c_in_per_group = w_shape[1];
        let kh = w_shape[2];
        let kw = w_shape[3];

        let group = node.attrs.get_int("group").unwrap_or(1) as usize;
        let strides_attr = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
        let pads_attr = node
            .attrs
            .get_ints("pads")
            .unwrap_or_else(|| vec![0, 0, 0, 0]);
        let dilations_attr = node
            .attrs
            .get_ints("dilations")
            .unwrap_or_else(|| vec![1, 1]);
        let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();

        let stride = Stride2D {
            h: strides_attr[0] as u32,
            w: strides_attr[1] as u32,
        };
        let dilation = Dilation2D {
            h: dilations_attr[0] as u32,
            w: dilations_attr[1] as u32,
        };

        let is_depthwise = group > 1 && c_in_per_group == 1;
        let c_out_per_group = c_out / group;

        // Input is NHWC: [N, H, W, C] (graph_opt inserted transposes)
        let in_shape = shape_map.get(input_name).cloned().unwrap_or_default();
        let (h_in, w_in) = if in_shape.len() == 4 {
            (in_shape[1], in_shape[2])
        } else {
            (1, 1)
        };

        let pad = compute_padding(
            &auto_pad, h_in, w_in, kh, kw, &stride, &dilation, &pads_attr,
        );

        // Input value (already NHWC from graph_opt Transpose)
        let input_id = self.get_or_define_value(input_name, shape_map)?;

        // Transpose weights OIHW → OHWI (or depthwise OIHW → HWGo) and define as static
        let weight_f = weight.floats().context("XNNPACK Conv weight")?;
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
                    bias_tensor.floats().context("XNNPACK Conv bias")?.to_vec(),
                )?
            } else {
                XNN_INVALID_VALUE_ID
            }
        } else {
            XNN_INVALID_VALUE_ID
        };

        // Output value (NHWC)
        let output_id = self.get_or_define_value(output_name, shape_map)?;

        let status = if is_depthwise {
            unsafe {
                xnn_define_depthwise_convolution_2d(
                    self.subgraph,
                    pad.top as u32,
                    pad.right as u32,
                    pad.bottom as u32,
                    pad.left as u32,
                    kh as u32,
                    kw as u32,
                    stride.h,
                    stride.w,
                    dilation.h,
                    dilation.w,
                    c_out_per_group as u32,
                    group,
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
                    pad.top as u32,
                    pad.right as u32,
                    pad.bottom as u32,
                    pad.left as u32,
                    kh as u32,
                    kw as u32,
                    stride.h,
                    stride.w,
                    dilation.h,
                    dilation.w,
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
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_convolution_2d failed: {status:?}");
        }
        Ok(())
    }

    fn add_maxpool(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let node = &cap.node;
        let ks_attr = node.attrs.get_ints("kernel_shape").unwrap_or_default();
        let strides_attr = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
        let pads_attr = node
            .attrs
            .get_ints("pads")
            .unwrap_or_else(|| vec![0, 0, 0, 0]);
        let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();

        let kh = ks_attr[0] as usize;
        let kw = ks_attr[1] as usize;
        let stride = Stride2D {
            h: strides_attr[0] as u32,
            w: strides_attr[1] as u32,
        };
        let dilation = Dilation2D { h: 1, w: 1 };

        // Input is NHWC: [N, H, W, C] (graph_opt inserted transposes)
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let (h_in, w_in) = if in_shape.len() == 4 {
            (in_shape[1], in_shape[2])
        } else {
            (1, 1)
        };

        let pad = compute_padding(
            &auto_pad, h_in, w_in, kh, kw, &stride, &dilation, &pads_attr,
        );

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;

        let status = unsafe {
            xnn_define_max_pooling_2d(
                self.subgraph,
                pad.top as u32,
                pad.right as u32,
                pad.bottom as u32,
                pad.left as u32,
                kh as u32,
                kw as u32,
                stride.h,
                stride.w,
                1,
                1,
                f32::NEG_INFINITY,
                f32::INFINITY,
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_max_pooling_2d failed: {status:?}");
        }
        Ok(())
    }

    fn add_average_pool(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let node = &cap.node;
        let ks_attr = node.attrs.get_ints("kernel_shape").unwrap_or_default();
        let strides_attr = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
        let pads_attr = node
            .attrs
            .get_ints("pads")
            .unwrap_or_else(|| vec![0, 0, 0, 0]);
        let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();

        let kh = ks_attr[0] as usize;
        let kw = ks_attr[1] as usize;
        let stride = Stride2D {
            h: strides_attr[0] as u32,
            w: strides_attr[1] as u32,
        };
        let dilation = Dilation2D { h: 1, w: 1 };

        // Input is NHWC: [N, H, W, C] (graph_opt inserted transposes)
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let (h_in, w_in) = if in_shape.len() == 4 {
            (in_shape[1], in_shape[2])
        } else {
            (1, 1)
        };

        let pad = compute_padding(
            &auto_pad, h_in, w_in, kh, kw, &stride, &dilation, &pads_attr,
        );

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;

        let status = unsafe {
            xnn_define_average_pooling_2d(
                self.subgraph,
                pad.top as u32,
                pad.right as u32,
                pad.bottom as u32,
                pad.left as u32,
                kh as u32,
                kw as u32,
                stride.h,
                stride.w,
                f32::NEG_INFINITY,
                f32::INFINITY,
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_average_pooling_2d failed: {status:?}");
        }
        Ok(())
    }

    fn add_global_avg_pool(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        // Input is NHWC: [N, H, W, C] (graph_opt inserted transposes)
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;

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
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_global_average_pooling_2d failed: {status:?}");
        }
        Ok(())
    }

    fn add_resize(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        // Input is NHWC: [N, H, W, C] (graph_opt inserted transposes)
        let output_name = &cap.outputs[0];
        let out_shape = shape_map.get(output_name).cloned().unwrap_or_default();
        if out_shape.len() != 4 {
            anyhow::bail!(
                "XNNPACK Resize: expected 4D output shape, got {:?}",
                out_shape
            );
        }
        let new_height = out_shape[1];
        let new_width = out_shape[2];

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(output_name, shape_map)?;

        let status = unsafe {
            xnn_define_static_resize_bilinear_2d(
                self.subgraph,
                new_height,
                new_width,
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_static_resize_bilinear_2d failed: {status:?}");
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Unary / binary ops (layout-agnostic)
    // -----------------------------------------------------------------------

    fn add_relu(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let params = xnn_unary_params {
            clamp: xnn_unary_params__bindgen_ty_1 {
                min: 0.0,
                max: f32::INFINITY,
            },
        };
        self.add_unary(
            cap,
            shape_map,
            xnn_unary_operator_xnn_unary_clamp,
            Some(params),
        )
    }

    fn add_clip(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let min_val = if cap.inputs.len() > 1 && !cap.inputs[1].is_empty() {
            initializers
                .get(&cap.inputs[1])
                .and_then(|t| t.floats().ok().map(|f| f[0]))
                .unwrap_or(f32::NEG_INFINITY)
        } else {
            cap.node.attrs.get_float("min").unwrap_or(f32::NEG_INFINITY)
        };
        let max_val = if cap.inputs.len() > 2 && !cap.inputs[2].is_empty() {
            initializers
                .get(&cap.inputs[2])
                .and_then(|t| t.floats().ok().map(|f| f[0]))
                .unwrap_or(f32::INFINITY)
        } else {
            cap.node.attrs.get_float("max").unwrap_or(f32::INFINITY)
        };
        let params = xnn_unary_params {
            clamp: xnn_unary_params__bindgen_ty_1 {
                min: min_val,
                max: max_val,
            },
        };
        self.add_unary(
            cap,
            shape_map,
            xnn_unary_operator_xnn_unary_clamp,
            Some(params),
        )
    }

    fn add_leaky_relu(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let alpha = cap.node.attrs.get_float("alpha").unwrap_or(0.01);
        let params = xnn_unary_params {
            leaky_relu: xnn_unary_params__bindgen_ty_3 {
                negative_slope: alpha,
            },
        };
        self.add_unary(
            cap,
            shape_map,
            xnn_unary_operator_xnn_unary_leaky_relu,
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
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let params_ptr = params
            .as_ref()
            .map(|p| p as *const xnn_unary_params)
            .unwrap_or(std::ptr::null());
        let status =
            unsafe { xnn_define_unary(self.subgraph, op_type, params_ptr, input_id, output_id, 0) };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_unary failed for {:?}: {status:?}", cap.op);
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
        let input2_id = self.get_or_define_value(&cap.inputs[1], shape_map)?;
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
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_binary failed for {:?}: {status:?}", cap.op);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // FC / matmul ops
    // -----------------------------------------------------------------------

    fn add_softmax(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe { xnn_define_softmax(self.subgraph, input_id, output_id, 0) };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_softmax failed: {status:?}");
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
        let trans_b = node.attrs.get_int("transB").unwrap_or(0) != 0;

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;

        let weight_name = &cap.inputs[1];
        let weight = initializers.get(weight_name).ok_or_else(|| {
            anyhow::anyhow!("XNNPACK Gemm: weight {weight_name} not in initializers")
        })?;
        let w_data = weight.floats().context("XNNPACK Gemm weight")?.to_vec();
        let w_shape: Vec<usize> = weight.dims.iter().copied().collect();
        let filter_id =
            self.define_static_value(&format!("{weight_name}__xnn"), &w_shape, w_data)?;

        let bias_id = if cap.inputs.len() > 2 && !cap.inputs[2].is_empty() {
            let bias_name = &cap.inputs[2];
            if let Some(bias_tensor) = initializers.get(bias_name) {
                self.define_static_value(
                    &format!("{bias_name}__xnn"),
                    &bias_tensor.dims.iter().copied().collect::<Vec<_>>(),
                    bias_tensor.floats().context("XNNPACK Gemm bias")?.to_vec(),
                )?
            } else {
                XNN_INVALID_VALUE_ID
            }
        } else {
            XNN_INVALID_VALUE_ID
        };

        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;

        // ONNX transB=1: weight is [O,I] — matches XNNPACK default
        // ONNX transB=0: weight is [I,O] — needs XNN_FLAG_TRANSPOSE_WEIGHTS
        let flags = if trans_b {
            0
        } else {
            XNN_FLAG_TRANSPOSE_WEIGHTS
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
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_fully_connected (Gemm) failed: {status:?}");
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
        let a_shape = shape_map.get(a_name).cloned().unwrap_or_default();
        let b_shape = shape_map.get(b_name).cloned().unwrap_or_default();

        // 2D static B → fully_connected
        if a_shape.len() == 2 && b_shape.len() == 2 && initializers.contains_key(b_name) {
            let input_id = self.get_or_define_value(a_name, shape_map)?;
            let weight = initializers.get(b_name).unwrap();
            let filter_id = self.define_static_value(
                &format!("{b_name}__xnn"),
                &b_shape,
                weight.floats().context("XNNPACK MatMul weight")?.to_vec(),
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
            if status != xnn_status_xnn_status_success {
                anyhow::bail!("xnn_define_fully_connected (MatMul) failed: {status:?}");
            }
        } else {
            let input1_id = self.get_or_define_value(a_name, shape_map)?;
            let input2_id = self.get_or_define_value(b_name, shape_map)?;
            let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
            let status = unsafe {
                xnn_define_batch_matrix_multiply(self.subgraph, input1_id, input2_id, output_id, 0)
            };
            if status != xnn_status_xnn_status_success {
                anyhow::bail!("xnn_define_batch_matrix_multiply failed: {status:?}");
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shape / layout ops
    // -----------------------------------------------------------------------

    fn add_flatten(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let axis = cap.node.attrs.get_int("axis").unwrap_or(1) as usize;
        let outer: usize = in_shape[..axis].iter().product();
        let inner: usize = in_shape[axis..].iter().product();
        let new_shape = [outer, inner];

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_static_reshape(self.subgraph, 2, new_shape.as_ptr(), input_id, output_id, 0)
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_static_reshape (Flatten) failed: {status:?}");
        }
        Ok(())
    }

    fn add_reshape(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let out_shape = shape_map.get(&cap.outputs[0]).cloned().unwrap_or_default();
        if out_shape.is_empty() || out_shape.len() > XNN_MAX_TENSOR_DIMS as usize {
            anyhow::bail!("XNNPACK Reshape: unknown or too many output dims");
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
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_static_reshape failed: {status:?}");
        }
        Ok(())
    }

    fn add_concat(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let raw_axis = cap.node.attrs.get_int("axis").unwrap_or(0);
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
                axis as i32,
                input_ids.len(),
                input_ids.as_ptr(),
                output_id,
                0,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_concatenate failed: {status:?}");
        }
        Ok(())
    }

    fn add_transpose(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let perm: Vec<usize> = if let Some(p) = cap.node.attrs.get_ints("perm") {
            p.iter().map(|&v| v as usize).collect()
        } else {
            (0..in_shape.len()).rev().collect()
        };
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_static_transpose(
                self.subgraph,
                perm.len(),
                perm.as_ptr(),
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_static_transpose failed: {status:?}");
        }
        Ok(())
    }

    fn add_identity(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe { xnn_define_copy(self.subgraph, input_id, output_id, 0) };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_copy (Identity) failed: {status:?}");
        }
        Ok(())
    }

    fn add_static_reshape(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let out_shape = shape_map.get(&cap.outputs[0]).cloned().unwrap_or_default();
        if out_shape.is_empty() || out_shape.len() > XNN_MAX_TENSOR_DIMS as usize {
            anyhow::bail!("XNNPACK {:?}: unknown or too many output dims", cap.op);
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
        if status != xnn_status_xnn_status_success {
            anyhow::bail!(
                "xnn_define_static_reshape ({:?}) failed: {status:?}",
                cap.op
            );
        }
        Ok(())
    }

    fn add_slice(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let out_shape = shape_map.get(&cap.outputs[0]).cloned().unwrap_or_default();
        if in_shape.is_empty() || out_shape.is_empty() {
            anyhow::bail!("XNNPACK Slice: unknown input or output dims");
        }
        let ndim = in_shape.len();
        let offsets = vec![0usize; ndim];
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe {
            xnn_define_static_slice(
                self.subgraph,
                ndim,
                offsets.as_ptr(),
                out_shape.as_ptr(),
                input_id,
                output_id,
                0,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_static_slice failed: {status:?}");
        }
        Ok(())
    }

    fn add_round(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe { xnn_define_bankers_rounding(self.subgraph, input_id, output_id, 0) };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_bankers_rounding failed: {status:?}");
        }
        Ok(())
    }

    fn add_cast(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let status = unsafe { xnn_define_convert(self.subgraph, input_id, output_id, 0) };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_convert (Cast) failed: {status:?}");
        }
        Ok(())
    }

    fn add_reduce_min(
        &mut self,
        cap: &CapturedOp,
        shape_map: &HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        let in_shape = shape_map.get(&cap.inputs[0]).cloned().unwrap_or_default();
        let raw_axes = cap.node.attrs.get_ints("axes").unwrap_or_default();
        let keepdims = cap.node.attrs.get_int("keepdims").unwrap_or(1) != 0;
        let ndim = in_shape.len() as i64;
        let axes: Vec<usize> = raw_axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (ndim + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        let input_id = self.get_or_define_value(&cap.inputs[0], shape_map)?;
        let output_id = self.get_or_define_value(&cap.outputs[0], shape_map)?;
        let mut flags = 0u32;
        if keepdims {
            flags |= XNN_FLAG_KEEP_DIMS;
        }
        let status = unsafe {
            xnn_define_static_reduce(
                self.subgraph,
                xnn_reduce_operator_xnn_reduce_min,
                axes.len(),
                axes.as_ptr(),
                input_id,
                output_id,
                flags,
            )
        };
        if status != xnn_status_xnn_status_success {
            anyhow::bail!("xnn_define_static_reduce (ReduceMin) failed: {status:?}");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Padding helper
// ---------------------------------------------------------------------------

fn compute_padding(
    auto_pad: &str,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: &Stride2D,
    dilation: &Dilation2D,
    pads_attr: &[i64],
) -> Padding2D {
    if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
        let eff_kh = dilation.h as usize * (kh - 1) + 1;
        let eff_kw = dilation.w as usize * (kw - 1) + 1;
        let oh = h_in.div_ceil(stride.h as usize);
        let ow = w_in.div_ceil(stride.w as usize);
        let pad_h = ((oh - 1) * stride.h as usize + eff_kh).saturating_sub(h_in);
        let pad_w = ((ow - 1) * stride.w as usize + eff_kw).saturating_sub(w_in);
        if auto_pad == "SAME_UPPER" {
            Padding2D {
                top: pad_h / 2,
                left: pad_w / 2,
                bottom: pad_h - pad_h / 2,
                right: pad_w - pad_w / 2,
            }
        } else {
            Padding2D {
                top: pad_h - pad_h / 2,
                left: pad_w - pad_w / 2,
                bottom: pad_h / 2,
                right: pad_w / 2,
            }
        }
    } else {
        Padding2D {
            top: pads_attr[0] as usize,
            left: pads_attr[1] as usize,
            bottom: pads_attr[2] as usize,
            right: pads_attr[3] as usize,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape propagation for the optimized (NHWC) graph
// ---------------------------------------------------------------------------

/// Propagate shapes forward through a sequence of ops.
/// External input shapes + initializer shapes must already be in the map.
/// For spatial ops, shapes are NHWC (graph-opt inserted transposes).
pub fn propagate_shapes(
    ops: &[CapturedOp],
    shape_map: &mut HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, Tensor>,
) {
    for op in ops {
        let shapes = infer_op_output_shapes(op, shape_map, initializers);
        for (name, shape) in shapes {
            // Only fill in missing shapes — don't overwrite shapes that are
            // already correct from layout-aware op_type.rs inference.
            shape_map.entry(name).or_insert(shape);
        }
    }
}

fn infer_op_output_shapes(
    op: &CapturedOp,
    shape_map: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, Tensor>,
) -> Vec<(String, Vec<usize>)> {
    let get =
        |idx: usize| -> Option<&Vec<usize>> { op.inputs.get(idx).and_then(|n| shape_map.get(n)) };

    let mut result = Vec::new();

    match op.op {
        OpType::Transpose | OpType::LayoutTranspose => {
            if let Some(x) = get(0) {
                let perm = op.node.attrs.get_ints("perm");
                let out = if let Some(p) = perm {
                    p.iter().map(|&i| x[i as usize]).collect()
                } else {
                    x.iter().rev().copied().collect()
                };
                result.push((op.outputs[0].clone(), out));
            }
        }

        OpType::Conv => {
            // Input is NHWC: [N, H, W, C_in] (graph_opt inserted transposes)
            if let (Some(x), Some(w_name)) = (get(0), op.inputs.get(1)) {
                if let Some(w) = initializers.get(w_name.as_str()) {
                    if x.len() == 4 && w.dims.len() == 4 {
                        let n = x[0];
                        let h_in = x[1];
                        let w_in = x[2];
                        let c_out = w.dims[0]; // weights are still OIHW
                        let kh = w.dims[2];
                        let kw = w.dims[3];
                        let strides = op
                            .node
                            .attrs
                            .get_ints("strides")
                            .unwrap_or_else(|| vec![1, 1]);
                        let dilations = op
                            .node
                            .attrs
                            .get_ints("dilations")
                            .unwrap_or_else(|| vec![1, 1]);
                        let pads = op
                            .node
                            .attrs
                            .get_ints("pads")
                            .unwrap_or_else(|| vec![0, 0, 0, 0]);
                        let auto_pad = op.node.attrs.get_string("auto_pad").unwrap_or_default();

                        let h_out = compute_spatial_out(
                            h_in,
                            kh,
                            strides[0] as usize,
                            dilations[0] as usize,
                            pads[0] as usize + pads[2] as usize,
                            &auto_pad,
                        );
                        let w_out = compute_spatial_out(
                            w_in,
                            kw,
                            strides[1] as usize,
                            dilations[1] as usize,
                            pads[1] as usize + pads[3] as usize,
                            &auto_pad,
                        );
                        // Output is NHWC: [N, H_out, W_out, C_out]
                        result.push((op.outputs[0].clone(), vec![n, h_out, w_out, c_out]));
                    }
                }
            }
        }

        OpType::MaxPool | OpType::AveragePool => {
            // Input is NHWC: [N, H, W, C] (graph_opt inserted transposes)
            if let Some(x) = get(0) {
                if x.len() == 4 {
                    let ks = op.node.attrs.get_ints("kernel_shape").unwrap_or_default();
                    let strides = op
                        .node
                        .attrs
                        .get_ints("strides")
                        .unwrap_or_else(|| vec![1, 1]);
                    let pads = op
                        .node
                        .attrs
                        .get_ints("pads")
                        .unwrap_or_else(|| vec![0, 0, 0, 0]);
                    let auto_pad = op.node.attrs.get_string("auto_pad").unwrap_or_default();
                    if ks.len() >= 2 {
                        let h_out = compute_spatial_out(
                            x[1],
                            ks[0] as usize,
                            strides[0] as usize,
                            1,
                            pads[0] as usize + pads[2] as usize,
                            &auto_pad,
                        );
                        let w_out = compute_spatial_out(
                            x[2],
                            ks[1] as usize,
                            strides[1] as usize,
                            1,
                            pads[1] as usize + pads[3] as usize,
                            &auto_pad,
                        );
                        // Output is NHWC: [N, H_out, W_out, C]
                        result.push((op.outputs[0].clone(), vec![x[0], h_out, w_out, x[3]]));
                    }
                }
            }
        }

        OpType::GlobalAveragePool => {
            // Input is NHWC: [N, H, W, C] → output is NHWC: [N, 1, 1, C]
            if let Some(x) = get(0) {
                if x.len() == 4 {
                    result.push((op.outputs[0].clone(), vec![x[0], 1, 1, x[3]]));
                }
            }
        }

        OpType::Resize => {
            // If output shape is already known, skip
            // Otherwise we'd need to read scales/sizes from initializers — complex
            // For now, rely on shape_map being populated from outside
        }

        // Elementwise unary — same shape as input
        OpType::Relu
        | OpType::Clip
        | OpType::Sigmoid
        | OpType::Tanh
        | OpType::Exp
        | OpType::Abs
        | OpType::Sqrt
        | OpType::Floor
        | OpType::Ceil
        | OpType::Log
        | OpType::Neg
        | OpType::Sin
        | OpType::Cos
        | OpType::LeakyRelu
        | OpType::Round
        | OpType::Cast
        | OpType::Identity => {
            if let Some(x) = get(0) {
                result.push((op.outputs[0].clone(), x.clone()));
            }
        }

        // Binary — broadcast shapes
        OpType::Add | OpType::Sub | OpType::Mul | OpType::Div | OpType::Max | OpType::Min => {
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                if let Some(out) = broadcast_shapes(a, b) {
                    result.push((op.outputs[0].clone(), out));
                }
            }
        }

        OpType::Softmax => {
            if let Some(x) = get(0) {
                result.push((op.outputs[0].clone(), x.clone()));
            }
        }

        OpType::Gemm => {
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                if a.len() == 2 && b.len() == 2 {
                    let trans_b = op.node.attrs.get_int("transB").unwrap_or(0) != 0;
                    let n = if trans_b { b[0] } else { b[1] };
                    result.push((op.outputs[0].clone(), vec![a[0], n]));
                }
            }
        }

        OpType::MatMul => {
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                if a.len() == 2 && b.len() == 2 {
                    result.push((op.outputs[0].clone(), vec![a[0], b[1]]));
                }
            }
        }

        OpType::Flatten => {
            if let Some(x) = get(0) {
                let axis = op.node.attrs.get_int("axis").unwrap_or(1) as usize;
                let outer: usize = x[..axis].iter().product();
                let inner: usize = x[axis..].iter().product();
                result.push((op.outputs[0].clone(), vec![outer, inner]));
            }
        }

        OpType::Reshape => {
            // Output shape comes from the shape input (initializer or constant-folded)
            if let Some(shape_name) = op.inputs.get(1) {
                if let Some(t) = initializers.get(shape_name.as_str()) {
                    if let Ok(vals) = t.ints() {
                        let in_shape = get(0);
                        let total: usize = in_shape.map(|s| s.iter().product()).unwrap_or(0);
                        let mut dims: Vec<usize> = Vec::new();
                        let mut infer_idx = None;
                        for (i, &v) in vals.iter().enumerate() {
                            if v == -1 {
                                infer_idx = Some(i);
                                dims.push(0);
                            } else if v == 0 {
                                dims.push(
                                    in_shape
                                        .map(|s| s.get(i).copied().unwrap_or(0))
                                        .unwrap_or(0),
                                );
                            } else {
                                dims.push(v as usize);
                            }
                        }
                        if let Some(idx) = infer_idx {
                            let known: usize = dims
                                .iter()
                                .enumerate()
                                .filter(|&(i, _)| i != idx)
                                .map(|(_, &d)| d)
                                .product();
                            if known > 0 {
                                dims[idx] = total / known;
                            }
                        }
                        result.push((op.outputs[0].clone(), dims));
                    }
                }
            }
        }

        OpType::Concat => {
            let raw_axis = op.node.attrs.get_int("axis").unwrap_or(0);
            if let Some(first) = get(0) {
                let ndim = first.len() as i64;
                let axis = (if raw_axis < 0 {
                    raw_axis + ndim
                } else {
                    raw_axis
                }) as usize;
                let mut out = first.clone();
                for i in 1..op.inputs.len() {
                    if let Some(s) = get(i) {
                        out[axis] += s[axis];
                    }
                }
                result.push((op.outputs[0].clone(), out));
            }
        }

        OpType::Unsqueeze => {
            if let Some(x) = get(0) {
                let axes = op.node.attrs.get_ints("axes").unwrap_or_default();
                let out_len = x.len() + axes.len();
                let mut out = vec![0usize; out_len];
                let axes_set: HashSet<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (out_len as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                let mut src = 0;
                for (i, out_val) in out.iter_mut().enumerate().take(out_len) {
                    if axes_set.contains(&i) {
                        *out_val = 1;
                    } else {
                        *out_val = x[src];
                        src += 1;
                    }
                }
                result.push((op.outputs[0].clone(), out));
            }
        }

        OpType::Squeeze => {
            if let Some(x) = get(0) {
                let axes = op.node.attrs.get_ints("axes").unwrap_or_default();
                let out: Vec<usize> = if axes.is_empty() {
                    x.iter().copied().filter(|&d| d != 1).collect()
                } else {
                    let ndim = x.len() as i64;
                    let axes_set: HashSet<usize> = axes
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (ndim + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect();
                    x.iter()
                        .enumerate()
                        .filter(|(i, _)| !axes_set.contains(i))
                        .map(|(_, &d)| d)
                        .collect()
                };
                result.push((op.outputs[0].clone(), out));
            }
        }

        OpType::ReduceMin => {
            if let Some(x) = get(0) {
                let axes = op.node.attrs.get_ints("axes").unwrap_or_default();
                let keepdims = op.node.attrs.get_int("keepdims").unwrap_or(1) != 0;
                let ndim = x.len() as i64;
                let axes_set: HashSet<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (ndim + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                let out: Vec<usize> = x
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &d)| {
                        if axes_set.contains(&i) {
                            if keepdims { Some(1) } else { None }
                        } else {
                            Some(d)
                        }
                    })
                    .collect();
                result.push((op.outputs[0].clone(), out));
            }
        }

        _ => {}
    }

    result
}

fn compute_spatial_out(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    total_pad: usize,
    auto_pad: &str,
) -> usize {
    match auto_pad {
        "SAME_UPPER" | "SAME_LOWER" => in_dim.div_ceil(stride),
        "VALID" => {
            let ek = dilation * (kernel - 1) + 1;
            (in_dim.saturating_sub(ek)) / stride + 1
        }
        _ => {
            let ek = dilation * (kernel - 1) + 1;
            (in_dim + total_pad - ek) / stride + 1
        }
    }
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut out = vec![0usize; len];
    for i in 0..len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        out[len - 1 - i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// CompiledSubgraph — holds the XNNPACK runtime
// ---------------------------------------------------------------------------

struct CompiledSubgraph {
    runtime: xnn_runtime_t,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
    _static_data: Vec<Vec<f32>>,
    _static_data_bytes: Vec<Vec<u8>>,
    /// Whether xnn_reshape_runtime has been called.
    /// Only needs to be called once since shapes are static after compilation.
    is_setup: bool,
}

unsafe impl Send for CompiledSubgraph {}

impl Drop for CompiledSubgraph {
    fn drop(&mut self) {
        if !self.runtime.is_null() {
            unsafe { xnn_delete_runtime(self.runtime) };
        }
    }
}

// ---------------------------------------------------------------------------
// XnnpackSubgraph — public API
// ---------------------------------------------------------------------------

/// A compiled XNNPACK subgraph that replaces a sequence of ONNX ops.
///
/// Unlike the old implementation, this does NOT handle NCHW↔NHWC conversion
/// internally. The graph-opt pass has already inserted Transpose nodes for
/// layout conversion. This module just maps ops 1:1 to XNNPACK API calls.
pub struct XnnpackSubgraph {
    compiled: Option<CompiledSubgraph>,
    compile_failed: bool,
    pub ops: Vec<CapturedOp>,
    required_outputs: Vec<String>,
    initializers: HashMap<String, Tensor>,
    shape_hints: HashMap<String, Vec<usize>>,
}

unsafe impl Send for XnnpackSubgraph {}

impl XnnpackSubgraph {
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
        }
    }

    pub fn output_names(&self) -> &[String] {
        &self.required_outputs
    }

    pub fn execute(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let _span = tracing::trace_span!("xnnpack_subgraph").entered();

        self.ensure_compiled(values)?;

        if self.compile_failed {
            return self.execute_fallback(values);
        }

        let compiled = self.compiled.as_ref().unwrap();

        // Verify runtime input shapes match compiled shapes.
        // On shape mismatch, recompile with the new shapes.
        let mut needs_recompile = false;
        for (i, name) in compiled.input_names.iter().enumerate() {
            if let Some(tensor) = values.get(name) {
                if tensor.dtype() != crate::DType::Float {
                    tracing::debug!("XNNPACK: dtype mismatch for '{name}'");
                    self.compiled = None;
                    self.compile_failed = true;
                    return self.execute_fallback(values);
                }
                let runtime_shape: Vec<usize> = tensor.dims.to_vec();
                if runtime_shape != compiled.input_shapes[i] {
                    tracing::debug!(
                        "XNNPACK: shape changed for '{name}': {:?} -> {:?}, recompiling",
                        compiled.input_shapes[i],
                        runtime_shape,
                    );
                    needs_recompile = true;
                    break;
                }
            }
        }
        if needs_recompile {
            // Drop old runtime and recompile with current shapes
            self.compiled = None;
            self.compile_failed = false;
            // Update shape hints from current runtime values
            for op in &self.ops {
                for name in op.inputs.iter().chain(op.outputs.iter()) {
                    if !name.is_empty() {
                        if let Some(t) = values.get(name) {
                            if !t.dims.is_empty() {
                                self.shape_hints.insert(name.clone(), t.dims.to_vec());
                            }
                        }
                    }
                }
            }
            self.ensure_compiled(values)?;
            if self.compile_failed {
                return self.execute_fallback(values);
            }
        }

        let compiled = self.compiled.as_mut().unwrap();
        let xnn_pad = (XNN_EXTRA_BYTES as usize).div_ceil(4); // padding in f32 elements

        // Ensure input tensors have XNN_EXTRA_BYTES padding and collect pointers.
        // Ensure output tensors are pre-allocated with padding.
        // We call setup every time since tensor buffer pointers may have
        // changed (other ops may have reallocated them).
        let num_external = compiled.input_names.len() + compiled.output_names.len();
        let mut external_values = Vec::with_capacity(num_external);

        for (i, name) in compiled.input_names.iter().enumerate() {
            let tensor = values
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("XNNPACK: missing input {name}"))?;
            let numel: usize = compiled.input_shapes[i].iter().product();
            // Ensure buffer has XNN_EXTRA_BYTES padding beyond the actual data
            let buf = tensor.as_mut_f32(numel + xnn_pad);
            // Zero the padding (XNNPACK may read it)
            for v in &mut buf[numel..] {
                *v = 0.0;
            }
            external_values.push(xnn_external_value {
                id: i as u32,
                data: buf.as_mut_ptr() as *mut c_void,
            });
        }

        for (i, name) in compiled.output_names.iter().enumerate() {
            let ext_id = (compiled.input_names.len() + i) as u32;
            let numel: usize = compiled.output_shapes[i].iter().product();
            let tensor = values.entry(name.clone()).or_default();
            let buf = tensor.as_mut_f32(numel + xnn_pad);
            external_values.push(xnn_external_value {
                id: ext_id,
                data: buf.as_mut_ptr() as *mut c_void,
            });
        }

        if !compiled.is_setup {
            let status = unsafe { xnn_reshape_runtime(compiled.runtime) };
            if status != xnn_status_xnn_status_success {
                tracing::debug!("XNNPACK: reshape failed: {:?}", status);
                self.compiled = None;
                self.compile_failed = true;
                return self.execute_fallback(values);
            }
            compiled.is_setup = true;
        }

        let status = unsafe {
            xnn_setup_runtime_v2(
                compiled.runtime,
                external_values.len(),
                external_values.as_ptr(),
            )
        };
        if status != xnn_status_xnn_status_success {
            tracing::debug!("XNNPACK: setup failed: {:?}", status);
            self.compiled = None;
            self.compile_failed = true;
            return self.execute_fallback(values);
        }

        let status = unsafe { xnn_invoke_runtime(compiled.runtime) };
        if status != xnn_status_xnn_status_success {
            tracing::debug!("XNNPACK: invoke failed: {:?}", status);
            self.compiled = None;
            self.compile_failed = true;
            return self.execute_fallback(values);
        }

        // Truncate padding from input tensors (restore original length)
        for (i, name) in compiled.input_names.iter().enumerate() {
            let numel: usize = compiled.input_shapes[i].iter().product();
            if let Some(tensor) = values.get_mut(name) {
                tensor.as_mut_f32(numel).truncate(numel);
            }
        }

        // Set correct dims and truncate padding on output tensors
        for (i, name) in compiled.output_names.iter().enumerate() {
            let shape = &compiled.output_shapes[i];
            let numel: usize = shape.iter().product();
            if let Some(tensor) = values.get_mut(name) {
                tensor.as_mut_f32(numel).truncate(numel);
                tensor.set_dims(shape);
            }
        }

        Ok(())
    }

    fn ensure_compiled(&mut self, values: &HashMap<String, Tensor>) -> Result<()> {
        if self.compiled.is_some() || self.compile_failed {
            return Ok(());
        }

        // Build shape_map from hints + initializers + runtime values
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

        // Check for non-float external inputs
        {
            let mut produced: HashSet<&str> = HashSet::new();
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
                                tracing::debug!(
                                    "XNNPACK: non-float input '{inp}' dtype={:?}",
                                    t.dtype()
                                );
                                self.compile_failed = true;
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }

        // Fill in any missing shapes (e.g. for Reshape outputs not in hints)
        propagate_shapes(&self.ops, &mut shape_map, &self.initializers);

        // Verify all shapes are known
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
            tracing::debug!("XNNPACK: missing shapes, falling back");
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
                    "XNNPACK: compiled subgraph: {} inputs, {} outputs, {} ops",
                    compiled.input_names.len(),
                    compiled.output_names.len(),
                    self.ops.len(),
                );
                self.compiled = Some(compiled);
                Ok(())
            }
            Err(e) => {
                tracing::debug!("XNNPACK: compile failed: {e:#}");
                self.compile_failed = true;
                Ok(())
            }
        }
    }

    fn execute_fallback(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        // Run each captured op on CPU via the normal plan execution path.
        // Load subgraph-local initializers into values so ops can find them.
        for (k, v) in &self.initializers {
            values.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for op in &self.ops {
            super::execute_node(&op.node, values).with_context(|| {
                format!(
                    "XNNPACK CPU fallback: {:?} {:?} -> {:?}",
                    op.op, op.inputs, op.outputs
                )
            })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// compile_subgraph — builds XNNPACK runtime from captured ops
// ---------------------------------------------------------------------------

fn compile_subgraph(
    ops: &[CapturedOp],
    required_outputs: &[String],
    shape_map: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, Tensor>,
) -> Result<CompiledSubgraph> {
    // Identify external inputs (consumed but not produced, not initializers)
    let mut produced: HashSet<String> = HashSet::new();
    let init_names: HashSet<&str> = initializers.keys().map(|s| s.as_str()).collect();
    for op in ops {
        for out in &op.outputs {
            produced.insert(out.clone());
        }
    }
    let mut external_input_set: HashSet<String> = HashSet::new();
    for op in ops {
        for inp in &op.inputs {
            if !inp.is_empty() && !produced.contains(inp) && !init_names.contains(inp.as_str()) {
                external_input_set.insert(inp.clone());
            }
        }
    }
    let external_inputs: Vec<String> = external_input_set.into_iter().collect();

    let num_external = (external_inputs.len() + required_outputs.len()) as u32;
    let mut builder = SubgraphBuilder::new(num_external)?;

    const MAX_BUF_ELEMS: usize = 256 * 1024 * 1024 / 4; // 256 MB cap

    // Define external inputs
    let mut input_shapes = Vec::new();
    for (i, name) in external_inputs.iter().enumerate() {
        let shape = shape_map.get(name).cloned().unwrap_or_default();
        let numel: usize = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .unwrap_or(0);
        if numel == 0 || numel > MAX_BUF_ELEMS {
            anyhow::bail!(
                "XNNPACK: input '{name}' has unreasonable shape {shape:?} (numel={numel})"
            );
        }
        builder.define_external_input(name, i as u32, &shape)?;
        input_shapes.push(shape);
    }

    // Define external outputs
    let mut output_shapes = Vec::new();
    for (i, name) in required_outputs.iter().enumerate() {
        let shape = shape_map.get(name).cloned().unwrap_or_default();
        let numel: usize = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .unwrap_or(0);
        if numel == 0 || numel > MAX_BUF_ELEMS {
            anyhow::bail!(
                "XNNPACK: output '{name}' has unreasonable shape {shape:?} (numel={numel})"
            );
        }
        let ext_id = (external_inputs.len() + i) as u32;
        builder.define_external_output(name, ext_id, &shape)?;
        output_shapes.push(shape);
    }

    // Pre-define initializers as static values
    for op in ops {
        for inp in &op.inputs {
            if !inp.is_empty() && !builder.value_ids.contains_key(inp) && !produced.contains(inp) {
                if let Some(tensor) = initializers.get(inp) {
                    if tensor.dtype() == crate::DType::Float {
                        let shape: Vec<usize> = tensor.dims.iter().copied().collect();
                        let data = tensor.floats().context("XNNPACK initializer")?.to_vec();
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

    if status != xnn_status_xnn_status_success {
        anyhow::bail!("xnn_create_runtime failed: {status:?}");
    }

    Ok(CompiledSubgraph {
        runtime,
        input_names: external_inputs,
        output_names: required_outputs.to_vec(),
        input_shapes,
        output_shapes,
        _static_data: builder.static_data,
        _static_data_bytes: builder.static_data_bytes,
        is_setup: false,
    })
}
