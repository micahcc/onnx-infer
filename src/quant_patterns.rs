//! Quantization pattern recognition for XNNPACK.
//!
//! Scans an ONNX graph for quantization subgraph patterns that XNNPACK can
//! accelerate natively using its quantized tensor and operator APIs:
//!
//! 1. **QDQ chains**: `DequantizeLinear → Op → QuantizeLinear` — the standard
//!    ONNX quantization format where float ops are bracketed by quantize/
//!    dequantize nodes. XNNPACK can execute the inner op directly on quantized
//!    data, eliminating the DQ/Q overhead.
//!
//! 2. **QLinear fused ops**: `QLinearConv`, `QLinearMatMul`, `QLinearAdd`,
//!    `QLinearGlobalAveragePool` — already-fused quantized ops that map 1:1
//!    to XNNPACK quantized operators.
//!
//! The output is a [`QuantMap`] that associates tensor names with their
//! quantization parameters and identifies which graph nodes participate in
//! quantized subgraphs eligible for XNNPACK compilation.

use std::collections::{HashMap, HashSet};

use crate::Tensor;
use crate::layers::OpType;
use crate::onnx_ir::{Graph, Node, NodeOp};

// ---------------------------------------------------------------------------
// Quantization parameter types
// ---------------------------------------------------------------------------

/// Per-tensor quantization parameters (scale + zero_point).
#[derive(Debug, Clone)]
pub struct PerTensorQuant {
    pub scale: f32,
    pub zero_point: i32,
}

/// Per-channel (channelwise) quantization parameters.
#[derive(Debug, Clone)]
pub struct PerChannelQuant {
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
    pub channel_dim: usize,
}

/// How a tensor is quantized.
#[derive(Debug, Clone)]
pub enum TensorQuant {
    PerTensor(PerTensorQuant),
    PerChannel(PerChannelQuant),
}

impl TensorQuant {
    /// Return per-tensor scale (panics if per-channel).
    pub fn scale(&self) -> f32 {
        match self {
            TensorQuant::PerTensor(q) => q.scale,
            TensorQuant::PerChannel(_) => panic!("expected per-tensor quant"),
        }
    }

    /// Return per-tensor zero_point (panics if per-channel).
    pub fn zero_point(&self) -> i32 {
        match self {
            TensorQuant::PerTensor(q) => q.zero_point,
            TensorQuant::PerChannel(_) => panic!("expected per-tensor quant"),
        }
    }

    pub fn is_per_channel(&self) -> bool {
        matches!(self, TensorQuant::PerChannel(_))
    }
}

// ---------------------------------------------------------------------------
// Recognized quantized subgraph
// ---------------------------------------------------------------------------

/// A recognized quantized operation that XNNPACK can execute natively.
#[derive(Debug, Clone)]
pub struct QuantizedOp {
    /// The core compute op (Conv, MatMul, Add, GlobalAveragePool, etc.)
    pub op_type: OpType,
    /// The graph node for the core op (carries attrs like kernel_shape, strides, etc.)
    pub core_node: Node,
    /// Input tensor name (the quantized input, before DQ)
    pub input_name: String,
    /// Quantization params for the input tensor
    pub input_quant: TensorQuant,
    /// Output tensor name (the quantized output, after Q)
    pub output_name: String,
    /// Quantization params for the output tensor
    pub output_quant: TensorQuant,
    /// Weight tensor name + quant params (for conv/matmul/FC)
    pub weight_quant: Option<(String, TensorQuant)>,
    /// Bias tensor name (if present)
    pub bias_name: Option<String>,
    /// All node indices from the original graph that this fused op replaces
    pub absorbed_node_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// QuantMap — the main output of pattern recognition
// ---------------------------------------------------------------------------

/// Maps tensor names to their quantization parameters, and tracks which
/// graph nodes are part of quantized subgraphs eligible for XNNPACK.
#[derive(Debug, Clone, Default)]
pub struct QuantMap {
    /// Quantization parameters for individual tensors.
    pub tensor_quant: HashMap<String, TensorQuant>,
    /// Recognized quantized ops that can be compiled to XNNPACK.
    pub quantized_ops: Vec<QuantizedOp>,
    /// Set of node indices that are absorbed into quantized ops
    /// (should be skipped during normal plan building for XNNPACK paths).
    pub absorbed_indices: HashSet<usize>,
}

// ---------------------------------------------------------------------------
// Pattern recognition entry point
// ---------------------------------------------------------------------------

/// Scan the graph for quantization patterns that XNNPACK can accelerate.
///
/// Call this after `graph_opt::optimize()` but before plan building.
/// Returns a `QuantMap` that the XNNPACK subgraph compiler uses to:
/// - Define quantized tensors (with scale/zero_point)
/// - Skip DQ/Q wrapper nodes (absorbed into XNNPACK quantized ops)
/// - Map QLinear fused ops to XNNPACK quantized operators
pub fn recognize_quant_patterns(
    graph: &Graph,
    initializers: &HashMap<String, Tensor>,
) -> QuantMap {
    let mut qmap = QuantMap::default();

    // Build lookup maps
    let producer_map = build_producer_map(&graph.nodes);
    let consumer_map = build_consumer_map(&graph.nodes);

    // Pass 1: Record quantization params from all DQ/Q nodes
    record_dq_q_params(graph, initializers, &mut qmap);

    // Pass 2: Recognize QDQ chains (DQ → Op → Q)
    recognize_qdq_chains(graph, initializers, &producer_map, &consumer_map, &mut qmap);

    // Pass 3: Recognize QLinear fused ops
    recognize_qlinear_ops(graph, initializers, &mut qmap);

    tracing::info!(
        "quant_patterns: {} quantized ops recognized, {} tensor quant params, {} absorbed nodes",
        qmap.quantized_ops.len(),
        qmap.tensor_quant.len(),
        qmap.absorbed_indices.len(),
    );

    qmap
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_producer_map(nodes: &[Node]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for output in &node.outputs {
            if !output.is_empty() {
                map.insert(output.clone(), i);
            }
        }
    }
    map
}

fn build_consumer_map(nodes: &[Node]) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            if !input.is_empty() {
                map.entry(input.clone()).or_default().push(i);
            }
        }
    }
    map
}

/// Extract scale and zero_point from initializers for a DQ or Q node.
fn extract_quant_params(
    node: &Node,
    initializers: &HashMap<String, Tensor>,
) -> Option<TensorQuant> {
    // inputs[1] = scale, inputs[2] = zero_point (optional)
    let scale_name = node.inputs.get(1)?;
    let scale_tensor = initializers.get(scale_name)?;
    let scale_f = scale_tensor.floats().ok()?;

    let zp_f: Vec<f32> = node
        .inputs
        .get(2)
        .filter(|s| !s.is_empty())
        .and_then(|name| initializers.get(name))
        .and_then(|t| t.floats().ok().map(|f| f.to_vec()))
        .unwrap_or_else(|| vec![0.0; scale_f.len()]);

    if scale_f.len() == 1 {
        Some(TensorQuant::PerTensor(PerTensorQuant {
            scale: scale_f[0],
            zero_point: zp_f[0].round() as i32,
        }))
    } else {
        let axis = match &node.op {
            NodeOp::DequantizeLinear { axis } => *axis as usize,
            _ => 0, // QuantizeLinear default axis
        };
        Some(TensorQuant::PerChannel(PerChannelQuant {
            scales: scale_f.to_vec(),
            zero_points: zp_f.iter().map(|z| z.round() as i32).collect(),
            channel_dim: axis,
        }))
    }
}

/// Record quantization params from all DQ and Q nodes into the QuantMap.
fn record_dq_q_params(
    graph: &Graph,
    initializers: &HashMap<String, Tensor>,
    qmap: &mut QuantMap,
) {
    for node in &graph.nodes {
        match node.op_type() {
            OpType::DequantizeLinear => {
                // The *input* to DQ is the quantized tensor
                if let Some(quant) = extract_quant_params(node, initializers) {
                    let quant_input = &node.inputs[0];
                    if !quant_input.is_empty() {
                        qmap.tensor_quant
                            .entry(quant_input.clone())
                            .or_insert(quant);
                    }
                }
            }
            OpType::QuantizeLinear => {
                // The *output* of Q is the quantized tensor
                if let Some(quant) = extract_quant_params(node, initializers) {
                    let quant_output = &node.outputs[0];
                    if !quant_output.is_empty() {
                        qmap.tensor_quant
                            .entry(quant_output.clone())
                            .or_insert(quant);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Ops that XNNPACK can execute in quantized mode.
fn is_xnnpack_quantizable(op: OpType) -> bool {
    matches!(
        op,
        OpType::Conv
            | OpType::MatMul
            | OpType::Gemm
            | OpType::Add
            | OpType::Sub
            | OpType::Mul
            | OpType::Relu
            | OpType::Clip
            | OpType::MaxPool
            | OpType::AveragePool
            | OpType::GlobalAveragePool
            | OpType::Sigmoid
            | OpType::Softmax
    )
}

/// Recognize DQ → Op → Q chains where the inner op can run quantized in XNNPACK.
///
/// Pattern: one or more DequantizeLinear nodes feed into an op, whose output
/// feeds into a single QuantizeLinear node. All DQ/Q scales and zero_points
/// must be known constants (initializers).
fn recognize_qdq_chains(
    graph: &Graph,
    initializers: &HashMap<String, Tensor>,
    producer_map: &HashMap<String, usize>,
    consumer_map: &HashMap<String, Vec<usize>>,
    qmap: &mut QuantMap,
) {
    for (core_idx, core_node) in graph.nodes.iter().enumerate() {
        let core_op = core_node.op_type();
        if !is_xnnpack_quantizable(core_op) {
            continue;
        }

        // Check: the primary input (inputs[0]) comes from a DequantizeLinear
        let input_dq_idx = match core_node.inputs.first() {
            Some(name) if !name.is_empty() => producer_map.get(name).copied(),
            _ => continue,
        };
        let input_dq_idx = match input_dq_idx {
            Some(idx) if graph.nodes[idx].op_type() == OpType::DequantizeLinear => idx,
            _ => continue,
        };
        let input_dq_node = &graph.nodes[input_dq_idx];

        // Extract input quantization params
        let input_quant = match extract_quant_params(input_dq_node, initializers) {
            Some(q) => q,
            None => continue,
        };

        // Check: the output feeds into exactly one QuantizeLinear
        let output_name = match core_node.outputs.first() {
            Some(name) if !name.is_empty() => name,
            _ => continue,
        };
        let output_q_indices = match consumer_map.get(output_name) {
            Some(indices) if indices.len() == 1 => indices,
            _ => continue, // skip if output has multiple consumers or none
        };
        let output_q_idx = output_q_indices[0];
        if graph.nodes[output_q_idx].op_type() != OpType::QuantizeLinear {
            continue;
        }
        let output_q_node = &graph.nodes[output_q_idx];

        // Extract output quantization params
        let output_quant = match extract_quant_params(output_q_node, initializers) {
            Some(q) => q,
            None => continue,
        };

        // For ops with weights (Conv, MatMul, Gemm), check if the weight also
        // comes from a DequantizeLinear (weight quantization)
        let (weight_quant, weight_dq_idx) = extract_weight_quant(
            core_op,
            core_node,
            &graph.nodes,
            initializers,
            producer_map,
        );

        // For Conv/Gemm, find the bias input
        let bias_name = extract_bias_name(core_op, core_node);

        // Build the list of absorbed node indices
        let mut absorbed = vec![input_dq_idx, core_idx, output_q_idx];
        if let Some(w_idx) = weight_dq_idx {
            absorbed.push(w_idx);
        }
        absorbed.sort();
        absorbed.dedup();

        let quantized_input = &input_dq_node.inputs[0];
        let quantized_output = &output_q_node.outputs[0];

        qmap.quantized_ops.push(QuantizedOp {
            op_type: core_op,
            core_node: core_node.clone(),
            input_name: quantized_input.clone(),
            input_quant,
            output_name: quantized_output.clone(),
            output_quant,
            weight_quant,
            bias_name,
            absorbed_node_indices: absorbed.clone(),
        });

        for idx in absorbed {
            qmap.absorbed_indices.insert(idx);
        }
    }
}

/// For weight-bearing ops (Conv, MatMul, Gemm), check if the weight input
/// comes from a DequantizeLinear and extract its quantization params.
fn extract_weight_quant(
    core_op: OpType,
    core_node: &Node,
    nodes: &[Node],
    initializers: &HashMap<String, Tensor>,
    producer_map: &HashMap<String, usize>,
) -> (Option<(String, TensorQuant)>, Option<usize>) {
    let weight_input_idx = match core_op {
        OpType::Conv | OpType::MatMul | OpType::Gemm => 1,
        _ => return (None, None),
    };

    let weight_name = match core_node.inputs.get(weight_input_idx) {
        Some(name) if !name.is_empty() => name,
        _ => return (None, None),
    };

    // Weight comes from DQ node → quantized weight
    if let Some(&dq_idx) = producer_map.get(weight_name) {
        let dq_node = &nodes[dq_idx];
        if dq_node.op_type() == OpType::DequantizeLinear {
            if let Some(quant) = extract_quant_params(dq_node, initializers) {
                let raw_weight_name = &dq_node.inputs[0];
                return (Some((raw_weight_name.clone(), quant)), Some(dq_idx));
            }
        }
    }

    (None, None)
}

/// Extract bias tensor name for Conv/Gemm ops.
fn extract_bias_name(core_op: OpType, core_node: &Node) -> Option<String> {
    match core_op {
        OpType::Conv | OpType::Gemm => core_node
            .inputs
            .get(2)
            .filter(|s| !s.is_empty())
            .cloned(),
        _ => None,
    }
}

/// Recognize QLinear fused ops and record their quantization params.
fn recognize_qlinear_ops(
    graph: &Graph,
    initializers: &HashMap<String, Tensor>,
    qmap: &mut QuantMap,
) {
    for (idx, node) in graph.nodes.iter().enumerate() {
        match node.op_type() {
            OpType::QLinearConv => {
                // Inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [bias]
                if let Some(qop) = recognize_qlinear_conv(idx, node, initializers) {
                    for i in &qop.absorbed_node_indices {
                        qmap.absorbed_indices.insert(*i);
                    }
                    // Record tensor quant params
                    qmap.tensor_quant
                        .entry(qop.input_name.clone())
                        .or_insert_with(|| qop.input_quant.clone());
                    qmap.tensor_quant
                        .entry(qop.output_name.clone())
                        .or_insert_with(|| qop.output_quant.clone());
                    qmap.quantized_ops.push(qop);
                }
            }
            OpType::QLinearMatMul => {
                // Inputs: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
                if let Some(qop) = recognize_qlinear_matmul(idx, node, initializers) {
                    for i in &qop.absorbed_node_indices {
                        qmap.absorbed_indices.insert(*i);
                    }
                    qmap.tensor_quant
                        .entry(qop.input_name.clone())
                        .or_insert_with(|| qop.input_quant.clone());
                    qmap.tensor_quant
                        .entry(qop.output_name.clone())
                        .or_insert_with(|| qop.output_quant.clone());
                    qmap.quantized_ops.push(qop);
                }
            }
            OpType::QLinearAdd => {
                // Inputs: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
                if let Some(qop) = recognize_qlinear_add(idx, node, initializers) {
                    for i in &qop.absorbed_node_indices {
                        qmap.absorbed_indices.insert(*i);
                    }
                    qmap.tensor_quant
                        .entry(qop.input_name.clone())
                        .or_insert_with(|| qop.input_quant.clone());
                    qmap.tensor_quant
                        .entry(qop.output_name.clone())
                        .or_insert_with(|| qop.output_quant.clone());
                    qmap.quantized_ops.push(qop);
                }
            }
            OpType::QLinearGlobalAveragePool => {
                // Inputs: x, x_scale, x_zp, y_scale, y_zp
                if let Some(qop) = recognize_qlinear_gap(idx, node, initializers) {
                    for i in &qop.absorbed_node_indices {
                        qmap.absorbed_indices.insert(*i);
                    }
                    qmap.tensor_quant
                        .entry(qop.input_name.clone())
                        .or_insert_with(|| qop.input_quant.clone());
                    qmap.tensor_quant
                        .entry(qop.output_name.clone())
                        .or_insert_with(|| qop.output_quant.clone());
                    qmap.quantized_ops.push(qop);
                }
            }
            _ => {}
        }
    }
}

fn extract_per_tensor_quant(
    scale_name: &str,
    zp_name: &str,
    initializers: &HashMap<String, Tensor>,
) -> Option<PerTensorQuant> {
    let scale = initializers.get(scale_name)?.floats().ok()?[0];
    let zp = if zp_name.is_empty() {
        0
    } else {
        initializers
            .get(zp_name)?
            .floats()
            .ok()?[0]
            .round() as i32
    };
    Some(PerTensorQuant {
        scale,
        zero_point: zp,
    })
}

fn extract_weight_quant_from_qlinear(
    w_name: &str,
    w_scale_name: &str,
    w_zp_name: &str,
    initializers: &HashMap<String, Tensor>,
) -> Option<(String, TensorQuant)> {
    let scale_t = initializers.get(w_scale_name)?;
    let scale_f = scale_t.floats().ok()?;

    if scale_f.len() == 1 {
        let zp = if w_zp_name.is_empty() {
            0
        } else {
            initializers
                .get(w_zp_name)?
                .floats()
                .ok()?[0]
                .round() as i32
        };
        Some((
            w_name.to_string(),
            TensorQuant::PerTensor(PerTensorQuant {
                scale: scale_f[0],
                zero_point: zp,
            }),
        ))
    } else {
        let zp_f: Vec<i32> = if w_zp_name.is_empty() {
            vec![0; scale_f.len()]
        } else {
            initializers
                .get(w_zp_name)?
                .floats()
                .ok()?
                .iter()
                .map(|z| z.round() as i32)
                .collect()
        };
        Some((
            w_name.to_string(),
            TensorQuant::PerChannel(PerChannelQuant {
                scales: scale_f.to_vec(),
                zero_points: zp_f,
                channel_dim: 0, // ONNX QLinear default: output channel dim
            }),
        ))
    }
}

/// QLinearConv: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [bias]
fn recognize_qlinear_conv(
    idx: usize,
    node: &Node,
    initializers: &HashMap<String, Tensor>,
) -> Option<QuantizedOp> {
    if node.inputs.len() < 8 {
        return None;
    }
    let x_name = &node.inputs[0];
    let x_quant = extract_per_tensor_quant(&node.inputs[1], &node.inputs[2], initializers)?;
    let w_quant = extract_weight_quant_from_qlinear(
        &node.inputs[3],
        &node.inputs[4],
        &node.inputs[5],
        initializers,
    )?;
    let y_quant = extract_per_tensor_quant(&node.inputs[6], &node.inputs[7], initializers)?;
    let bias_name = node
        .inputs
        .get(8)
        .filter(|s| !s.is_empty())
        .cloned();

    // Build a synthetic Conv node from the QLinearConv attrs
    let conv_node = Node {
        op: match &node.op {
            NodeOp::QLinearConv {
                kernel_shape,
                strides,
                pads,
                dilations,
                group,
                auto_pad,
            } => NodeOp::Conv {
                kernel_shape: kernel_shape.clone(),
                strides: strides.clone(),
                pads: pads.clone(),
                dilations: dilations.clone(),
                group: *group,
                auto_pad: auto_pad.clone(),
            },
            _ => return None,
        },
        name: node.name.clone(),
        // Remap inputs for the Conv: input, weight, [bias]
        inputs: {
            let mut v = vec![x_name.clone(), node.inputs[3].clone()];
            if let Some(ref b) = bias_name {
                v.push(b.clone());
            }
            v
        },
        outputs: node.outputs.clone(),
    };

    Some(QuantizedOp {
        op_type: OpType::Conv,
        core_node: conv_node,
        input_name: x_name.clone(),
        input_quant: TensorQuant::PerTensor(x_quant),
        output_name: node.outputs[0].clone(),
        output_quant: TensorQuant::PerTensor(y_quant),
        weight_quant: Some(w_quant),
        bias_name,
        absorbed_node_indices: vec![idx],
    })
}

/// QLinearMatMul: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
fn recognize_qlinear_matmul(
    idx: usize,
    node: &Node,
    initializers: &HashMap<String, Tensor>,
) -> Option<QuantizedOp> {
    if node.inputs.len() < 8 {
        return None;
    }
    let a_quant = extract_per_tensor_quant(&node.inputs[1], &node.inputs[2], initializers)?;
    let b_quant = extract_weight_quant_from_qlinear(
        &node.inputs[3],
        &node.inputs[4],
        &node.inputs[5],
        initializers,
    )?;
    let y_quant = extract_per_tensor_quant(&node.inputs[6], &node.inputs[7], initializers)?;

    let matmul_node = Node {
        op: NodeOp::MatMul,
        name: node.name.clone(),
        inputs: vec![node.inputs[0].clone(), node.inputs[3].clone()],
        outputs: node.outputs.clone(),
    };

    Some(QuantizedOp {
        op_type: OpType::MatMul,
        core_node: matmul_node,
        input_name: node.inputs[0].clone(),
        input_quant: TensorQuant::PerTensor(a_quant),
        output_name: node.outputs[0].clone(),
        output_quant: TensorQuant::PerTensor(y_quant),
        weight_quant: Some(b_quant),
        bias_name: None,
        absorbed_node_indices: vec![idx],
    })
}

/// QLinearAdd: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
fn recognize_qlinear_add(
    idx: usize,
    node: &Node,
    initializers: &HashMap<String, Tensor>,
) -> Option<QuantizedOp> {
    if node.inputs.len() < 8 {
        return None;
    }
    let a_quant = extract_per_tensor_quant(&node.inputs[1], &node.inputs[2], initializers)?;
    let _b_quant = extract_per_tensor_quant(&node.inputs[4], &node.inputs[5], initializers)?;
    let y_quant = extract_per_tensor_quant(&node.inputs[6], &node.inputs[7], initializers)?;

    let add_node = Node {
        op: NodeOp::Add {
            legacy_broadcast: false,
            axis: 0,
        },
        name: node.name.clone(),
        inputs: vec![node.inputs[0].clone(), node.inputs[3].clone()],
        outputs: node.outputs.clone(),
    };

    Some(QuantizedOp {
        op_type: OpType::Add,
        core_node: add_node,
        input_name: node.inputs[0].clone(),
        input_quant: TensorQuant::PerTensor(a_quant),
        output_name: node.outputs[0].clone(),
        output_quant: TensorQuant::PerTensor(y_quant),
        weight_quant: None, // second input quant handled via tensor_quant map
        bias_name: None,
        absorbed_node_indices: vec![idx],
    })
}

/// QLinearGlobalAveragePool: x, x_scale, x_zp, y_scale, y_zp
fn recognize_qlinear_gap(
    idx: usize,
    node: &Node,
    initializers: &HashMap<String, Tensor>,
) -> Option<QuantizedOp> {
    if node.inputs.len() < 5 {
        return None;
    }
    let x_quant = extract_per_tensor_quant(&node.inputs[1], &node.inputs[2], initializers)?;
    let y_quant = extract_per_tensor_quant(&node.inputs[3], &node.inputs[4], initializers)?;

    let gap_node = Node {
        op: NodeOp::GlobalAveragePool,
        name: node.name.clone(),
        inputs: vec![node.inputs[0].clone()],
        outputs: node.outputs.clone(),
    };

    Some(QuantizedOp {
        op_type: OpType::GlobalAveragePool,
        core_node: gap_node,
        input_name: node.inputs[0].clone(),
        input_quant: TensorQuant::PerTensor(x_quant),
        output_name: node.outputs[0].clone(),
        output_quant: TensorQuant::PerTensor(y_quant),
        weight_quant: None,
        bias_name: None,
        absorbed_node_indices: vec![idx],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::dims;

    fn make_scalar_tensor(val: f32) -> Tensor {
        Tensor::new(dims![], vec![val])
    }

    #[test]
    fn test_qdq_chain_recognition() {
        let mut initializers = HashMap::new();
        initializers.insert("x_scale".to_string(), make_scalar_tensor(0.5));
        initializers.insert("x_zp".to_string(), make_scalar_tensor(128.0));
        initializers.insert("y_scale".to_string(), make_scalar_tensor(0.25));
        initializers.insert("y_zp".to_string(), make_scalar_tensor(64.0));

        let nodes = vec![
            Node {
                op: NodeOp::DequantizeLinear { axis: 1 },
                name: "dq_input".to_string(),
                inputs: vec![
                    "x_quant".to_string(),
                    "x_scale".to_string(),
                    "x_zp".to_string(),
                ],
                outputs: vec!["x_float".to_string()],
            },
            Node {
                op: NodeOp::Relu,
                name: "relu".to_string(),
                inputs: vec!["x_float".to_string()],
                outputs: vec!["y_float".to_string()],
            },
            Node {
                op: NodeOp::QuantizeLinear,
                name: "q_output".to_string(),
                inputs: vec![
                    "y_float".to_string(),
                    "y_scale".to_string(),
                    "y_zp".to_string(),
                ],
                outputs: vec!["y_quant".to_string()],
            },
        ];

        let graph = Graph {
            nodes,
            inputs: vec![],
            outputs: vec![],
            initializers,
            opset_version: 13,
        };

        let qmap = recognize_quant_patterns(&graph, &graph.initializers);
        assert_eq!(qmap.quantized_ops.len(), 1);
        assert_eq!(qmap.quantized_ops[0].op_type, OpType::Relu);
        assert_eq!(qmap.quantized_ops[0].input_name, "x_quant");
        assert_eq!(qmap.quantized_ops[0].output_name, "y_quant");
        assert_eq!(qmap.absorbed_indices.len(), 3);
    }
}
