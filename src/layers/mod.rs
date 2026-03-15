pub mod add;
pub mod batch_norm;
pub mod cast;
pub mod ceil;
pub mod clip;
pub mod concat;
pub mod constant;
pub mod conv;
pub mod dequantize_linear;
pub mod div;
pub mod exp;
pub mod flatten;
pub mod gather;
pub mod gemm;
pub mod global_avg_pool;
pub mod identity;
pub mod leaky_relu;
pub mod loop_op;
pub mod matmul;
pub mod maxpool;
pub mod mul;
pub mod nms;
pub mod qlinear_add;
pub mod qlinear_conv;
pub mod qlinear_global_avg_pool;
pub mod qlinear_matmul;
pub mod quantize_linear;
pub mod reduce_min;
pub mod relu;
pub mod reshape;
pub mod resize;
pub mod round;
pub mod shape_op;
pub mod sigmoid;
pub mod slice;
pub mod softmax;
pub mod squeeze;
pub mod sub;
pub mod tile;
pub mod transpose;
pub mod unsqueeze;

use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape;
use crate::get_attr_float;
use crate::get_attr_int;
use crate::get_attr_ints;
use crate::get_attr_string;
use crate::onnx::NodeProto;

pub trait Layer {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()>;
}

// Shared helpers for quantize ops
pub fn dequantize(data: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter().map(|&v| (v - zero_point) * scale).collect()
}

pub fn quantize_u8(data: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter()
        .map(|&v| (v / scale + zero_point).round().clamp(0.0, 255.0))
        .collect()
}

/// Shared helper for binary ops. `a` and `b` must already be f32 (pre-cast).
pub fn binary_op(
    a: &Tensor,
    b: &Tensor,
    output: &mut Tensor,
    legacy_broadcast: bool,
    axis: usize,
    op: fn(f32, f32) -> f32,
) -> Result<()> {
    let b_dims = if legacy_broadcast && b.dims.len() < a.dims.len() {
        let mut new_dims = vec![1usize; a.dims.len()];
        for (i, &d) in b.dims.iter().enumerate() {
            new_dims[axis + i] = d;
        }
        new_dims
    } else {
        b.dims.clone()
    };

    let out_shape = broadcast_shape(&a.dims, &b_dims);
    let numel: usize = out_shape.iter().product();
    let buf = output.as_mut_f32(numel);

    let a_f = a.floats();
    let b_f = b.floats();
    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];

    for val in buf.iter_mut() {
        let ai = broadcast_index(&index, &a.dims, &out_shape);
        let bi = broadcast_index(&index, &b_dims, &out_shape);
        *val = op(a_f[ai], b_f[bi]);

        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }

    output.dims = out_shape;
    Ok(())
}

/// The plan step for a single-output layer.
pub struct Step {
    pub output: String,
    pub layer: Box<dyn Layer>,
}

/// Build the execution plan from an ONNX graph.
/// Returns (steps, loop_steps, initializers, output_names).
pub fn build_plan(
    graph: &crate::onnx::GraphProto,
) -> Result<(Vec<PlanNode>, HashMap<String, Tensor>, Vec<String>)> {
    let mut initializers = HashMap::new();
    for init in &graph.initializer {
        if !init.name.is_empty() {
            initializers.insert(init.name.clone(), Tensor::from_proto(init)?);
        }
    }

    let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    let mut plan = Vec::new();
    for node in &graph.node {
        plan.push(build_node(node)?);
    }

    Ok((plan, initializers, output_names))
}

pub enum PlanNode {
    Single {
        output: String,
        layer: Box<dyn Layer>,
    },
    Loop(loop_op::Loop),
}

fn build_node(node: &NodeProto) -> Result<PlanNode> {
    let inputs = node.input.clone();

    if node.op_type == "Loop" {
        let body = node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("Loop: no body graph".into()))?
            .clone();
        return Ok(PlanNode::Loop(loop_op::Loop::new(
            inputs,
            node.output.clone(),
            body,
        )));
    }

    let output = if node.output.is_empty() || node.output[0].is_empty() {
        String::new()
    } else {
        node.output[0].clone()
    };

    let layer: Box<dyn Layer> = match node.op_type.as_str() {
        "Relu" => Box::new(relu::Relu::new(inputs)),
        "LeakyRelu" => Box::new(leaky_relu::LeakyRelu::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(0.01),
        )),
        "Clip" => Box::new(clip::Clip::new(
            inputs,
            get_attr_float(node, "min").unwrap_or(f32::NEG_INFINITY),
            get_attr_float(node, "max").unwrap_or(f32::INFINITY),
        )),
        "BatchNormalization" => Box::new(batch_norm::BatchNorm::new(
            inputs,
            get_attr_float(node, "epsilon").unwrap_or(1e-5),
        )),
        "Sigmoid" => Box::new(sigmoid::Sigmoid::new(inputs)),
        "Exp" => Box::new(exp::Exp::new(inputs)),
        "Ceil" => Box::new(ceil::Ceil::new(inputs)),
        "Round" => Box::new(round::Round::new(inputs)),
        "Softmax" => Box::new(softmax::Softmax::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(-1),
        )),
        "Add" => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(add::Add::new(inputs, lb, axis))
        }
        "Sub" => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(sub::Sub::new(inputs, lb, axis))
        }
        "Mul" => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(mul::Mul::new(inputs, lb, axis))
        }
        "Div" => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(div::Div::new(inputs, lb, axis))
        }
        "Conv" => {
            let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
            let st = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pa = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
            let gr = get_attr_int(node, "group").unwrap_or(1) as usize;
            let ap = get_attr_string(node, "auto_pad").unwrap_or_default();
            Box::new(conv::Conv::new(inputs, ks, st, pa, di, gr, ap))
        }
        "MatMul" => Box::new(matmul::MatMul::new(inputs)),
        "Gemm" => Box::new(gemm::Gemm::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(1.0),
            get_attr_float(node, "beta").unwrap_or(1.0),
            get_attr_int(node, "transA").unwrap_or(0) != 0,
            get_attr_int(node, "transB").unwrap_or(0) != 0,
        )),
        "MaxPool" => {
            let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
            let st = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pa = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let ap = get_attr_string(node, "auto_pad").unwrap_or_default();
            Box::new(maxpool::MaxPool::new(inputs, ks, st, pa, ap)?)
        }
        "GlobalAveragePool" => Box::new(global_avg_pool::GlobalAvgPool::new(inputs)),
        "Flatten" => Box::new(flatten::Flatten::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(1) as usize,
        )),
        "Shape" => Box::new(shape_op::Shape::new(inputs)),
        "Gather" => Box::new(gather::Gather::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(0),
        )),
        "Unsqueeze" => Box::new(unsqueeze::Unsqueeze::new(inputs, node)),
        "Concat" => Box::new(concat::Concat::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(0),
        )),
        "Identity" => Box::new(identity::Identity::new(inputs)),
        "Cast" => Box::new(cast::Cast::new(
            inputs,
            get_attr_int(node, "to").unwrap_or(1),
        )),
        "Transpose" => {
            let perm = get_attr_ints(node, "perm").map(|p| p.iter().map(|&v| v as usize).collect());
            Box::new(transpose::Transpose::new(inputs, perm))
        }
        "Squeeze" => Box::new(squeeze::Squeeze::new(inputs, node)),
        "Slice" => Box::new(slice::Slice::new(inputs)),
        "Tile" => Box::new(tile::Tile::new(inputs)),
        "Resize" => Box::new(resize::Resize::new(inputs)),
        "Reshape" => Box::new(reshape::Reshape::new(inputs, node)),
        "Constant" => {
            let attr = node
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .ok_or_else(|| InferenceError::InvalidModel("Constant has no value".into()))?;
            let tensor_proto = attr.t.as_ref().ok_or_else(|| {
                InferenceError::InvalidModel("Constant value is not a tensor".into())
            })?;
            let tensor = Tensor::from_proto(tensor_proto)?;
            Box::new(constant::Constant::new(tensor))
        }
        "ReduceMin" => Box::new(reduce_min::ReduceMin::new(
            inputs,
            get_attr_int(node, "keepdims").unwrap_or(1) != 0,
            node,
        )),
        "NonMaxSuppression" => Box::new(nms::Nms::new(inputs)),
        "QuantizeLinear" => Box::new(quantize_linear::QuantizeLinear::new(inputs)),
        "DequantizeLinear" => Box::new(dequantize_linear::DequantizeLinear::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(1),
        )),
        "QLinearConv" => {
            let has_bias = node.input.len() > 8 && !node.input[8].is_empty();
            let mut conv_inputs = vec!["__qconv_x__".to_string(), "__qconv_w__".to_string()];
            if has_bias {
                conv_inputs.push("__qconv_b__".to_string());
            }
            let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
            let st = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pa = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
            let gr = get_attr_int(node, "group").unwrap_or(1) as usize;
            let ap = get_attr_string(node, "auto_pad").unwrap_or_default();
            let inner = conv::Conv::new(conv_inputs, ks, st, pa, di, gr, ap);
            Box::new(qlinear_conv::QLinearConv::new(inputs, inner))
        }
        "QLinearAdd" => Box::new(qlinear_add::QLinearAdd::new(inputs)),
        "QLinearMatMul" => {
            let inner = matmul::MatMul::new(vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()]);
            Box::new(qlinear_matmul::QLinearMatMul::new(inputs, inner))
        }
        "QLinearGlobalAveragePool" => {
            let inner = global_avg_pool::GlobalAvgPool::new(vec!["__qgap_x__".to_string()]);
            Box::new(qlinear_global_avg_pool::QLinearGlobalAvgPool::new(
                inputs, inner,
            ))
        }
        op => return Err(InferenceError::UnsupportedOperator(op.to_string())),
    };

    Ok(PlanNode::Single { output, layer })
}

/// Execute a single ONNX node using the dispatch mechanism.
/// Used by Loop's body execution which still works with raw NodeProto.
pub fn execute_node(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let _span = tracing::trace_span!("op", op = %node.op_type, name = %node.name).entered();

    if node.op_type == "Loop" {
        let body = node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("Loop: no body graph".into()))?
            .clone();
        let loop_layer = loop_op::Loop::new(node.input.clone(), node.output.clone(), body);
        return loop_layer.execute(values);
    }

    if node.output.is_empty() || node.output[0].is_empty() {
        return Ok(());
    }

    let mut plan_node = build_node(node)?;
    match &mut plan_node {
        PlanNode::Single { output, layer } => {
            let mut out = values.remove(output.as_str()).unwrap_or_default();
            let result = layer.execute(values, &mut out);
            values.insert(output.clone(), out);
            result
        }
        PlanNode::Loop(loop_layer) => loop_layer.execute(values),
    }
}
