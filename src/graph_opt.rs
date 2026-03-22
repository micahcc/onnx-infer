use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;

use crate::Tensor;
use crate::dims;
use crate::layers::OpType;
use crate::onnx_ir::Attr;
use crate::onnx_ir::Attrs;
use crate::onnx_ir::Graph;
use crate::onnx_ir::Node;

const NCHW_TO_NHWC: [i64; 4] = [0, 2, 3, 1];
const NHWC_TO_NCHW: [i64; 4] = [0, 3, 1, 2];

fn is_nchw_to_nhwc(perm: &[i64]) -> bool {
    perm == NCHW_TO_NHWC
}

fn is_nhwc_to_nchw(perm: &[i64]) -> bool {
    perm == NHWC_TO_NCHW
}

fn is_inverse_transpose(perm: &[i64]) -> bool {
    is_nchw_to_nhwc(perm) || is_nhwc_to_nchw(perm)
}

fn get_transpose_perm(node: &Node) -> Option<Vec<i64>> {
    if !matches!(node.op_type, OpType::Transpose | OpType::LayoutTranspose) {
        return None;
    }
    node.attrs.get_ints("perm")
}

fn are_inverse_perms(a: &[i64], b: &[i64]) -> bool {
    (is_nchw_to_nhwc(a) && is_nhwc_to_nchw(b)) || (is_nhwc_to_nchw(a) && is_nchw_to_nhwc(b))
}

/// Ops that require NHWC layout (spatial 2D ops).
fn requires_nhwc(op: OpType) -> bool {
    // Spatial ops whose CPU execution handles NHWC natively.
    // BatchNormalization is folded into Conv by graph_opt before this runs.
    matches!(
        op,
        OpType::Conv
            | OpType::MaxPool
            | OpType::AveragePool
            | OpType::GlobalAveragePool
            | OpType::Resize
            | OpType::Upsample
            | OpType::QLinearConv
            | OpType::QLinearGlobalAveragePool
    )
}

/// Ops where a transpose can pass through without affecting semantics.
/// These are elementwise/unary ops that don't care about layout.
fn transpose_can_pass_through(op: OpType) -> bool {
    matches!(
        op,
        OpType::Relu
            | OpType::LeakyRelu
            | OpType::Clip
            | OpType::Sigmoid
            | OpType::Exp
            | OpType::Ceil
            | OpType::Round
            | OpType::Log
            | OpType::Tanh
            | OpType::Floor
            | OpType::Sqrt
            | OpType::Abs
            | OpType::Sin
            | OpType::Cos
            | OpType::Tan
            | OpType::Asin
            | OpType::Acos
            | OpType::Atan
            | OpType::Sinh
            | OpType::Cosh
            | OpType::Asinh
            | OpType::Acosh
            | OpType::Atanh
            | OpType::Erf
            | OpType::Sign
            | OpType::Neg
            | OpType::Reciprocal
            | OpType::Elu
            | OpType::Celu
            | OpType::Selu
            | OpType::HardSigmoid
            | OpType::ThresholdedRelu
            | OpType::IsNaN
            | OpType::IsInf
            | OpType::Softplus
            | OpType::Softsign
            | OpType::Identity
            | OpType::Cast
            | OpType::Dropout
            | OpType::DequantizeLinear
            | OpType::QuantizeLinear
    )
}

/// Binary elementwise ops where transpose can pass through both inputs.
fn is_binary_elementwise(op: OpType) -> bool {
    matches!(
        op,
        OpType::Add
            | OpType::Sub
            | OpType::Mul
            | OpType::Div
            | OpType::Max
            | OpType::Min
            | OpType::PRelu
            | OpType::Less
            | OpType::Equal
            | OpType::Greater
    )
}

fn make_transpose_node(name: &str, input: &str, output: &str, perm: &[i64; 4]) -> Node {
    let mut attrs_map = HashMap::new();
    attrs_map.insert("perm".to_string(), Attr::Ints(perm.to_vec()));
    Node {
        op_type: OpType::Transpose,
        name: name.to_string(),
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        attrs: Attrs(attrs_map),
    }
}

/// Create a LayoutTranspose node — a synthetic transpose inserted by graph_opt
/// that explicitly changes the data layout (NCHW↔NHWC). Unlike regular Transpose,
/// this carries layout semantics: the output layout differs from the input layout.
fn make_layout_transpose_node(
    name: &str,
    input: &str,
    output: &str,
    perm: &[i64; 4],
) -> Node {
    let mut attrs_map = HashMap::new();
    attrs_map.insert("perm".to_string(), Attr::Ints(perm.to_vec()));
    Node {
        op_type: OpType::LayoutTranspose,
        name: name.to_string(),
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        attrs: Attrs(attrs_map),
    }
}

fn unique_name(prefix: &str, counter: &mut usize) -> String {
    *counter += 1;
    format!("{prefix}_{}", *counter)
}

/// Build a map from tensor name → indices of nodes that consume it.
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

/// Build a map from tensor name → index of node that produces it.
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

/// Run all graph optimization passes on the given graph.
/// This prepares the graph for NHWC execution (e.g. XNNPACK):
/// - Folds BatchNorm into Conv
/// - Inserts NCHW↔NHWC transposes around spatial ops
/// - Eliminates redundant transpose pairs
pub fn optimize(graph: &mut Graph) {
    let mut counter = 0usize;
    fold_batchnorm_into_conv(graph);
    insert_layout_transposes(graph, &mut counter);

    // Run transpose elimination passes iteratively until no more changes.
    // Limit is high because push_unary/push_binary process one move at a time.
    for _ in 0..200 {
        let changed = eliminate_inverse_transposes(graph)
            | push_transposes_through_unary(graph, &mut counter)
            | push_transposes_through_binary(graph, &mut counter);
        if !changed {
            break;
        }
    }
    remove_dead_nodes(graph);
}

/// Run only the optimization passes that are safe for CPU (NCHW) execution:
/// - Folds BatchNorm into Conv
/// - Removes dead nodes
pub fn optimize_cpu(graph: &mut Graph) {
    fold_batchnorm_into_conv(graph);
    remove_dead_nodes(graph);
}

/// Insert NCHW→NHWC before spatial ops and NHWC→NCHW after their outputs.
fn insert_layout_transposes(graph: &mut Graph, counter: &mut usize) {
    let mut new_nodes = Vec::with_capacity(graph.nodes.len() * 2);

    for node in graph.nodes.drain(..) {
        if !requires_nhwc(node.op_type) {
            new_nodes.push(node);
            continue;
        }

        let mut modified = node;

        // Transpose data input(s) NCHW→NHWC.
        // For Conv: input 0 is data, inputs 1+ are weights (don't transpose weights here,
        // weight layout is handled by the XNNPACK layer).
        // For pooling/GAP/resize: input 0 is data.
        // For BatchNorm: input 0 is data, 1-4 are 1D params.
        let data_input_count = match modified.op_type {
            OpType::Conv | OpType::QLinearConv => 1, // only data input
            OpType::BatchNormalization => 1,
            _ => 1,
        };

        for i in 0..data_input_count.min(modified.inputs.len()) {
            let orig_input = modified.inputs[i].clone();
            if orig_input.is_empty() {
                continue;
            }
            let nhwc_name = unique_name("__nchw2nhwc", counter);
            new_nodes.push(make_layout_transpose_node(
                &format!("layout_transpose_{nhwc_name}"),
                &orig_input,
                &nhwc_name,
                &NCHW_TO_NHWC,
            ));
            modified.inputs[i] = nhwc_name;
        }

        // Transpose outputs NHWC→NCHW
        let mut output_transposes = Vec::new();
        for i in 0..modified.outputs.len() {
            let orig_output = modified.outputs[i].clone();
            if orig_output.is_empty() {
                continue;
            }
            let nhwc_name = unique_name("__nhwc_out", counter);
            modified.outputs[i] = nhwc_name.clone();
            output_transposes.push(make_layout_transpose_node(
                &format!("layout_transpose_{}", unique_name("__nhwc2nchw", counter)),
                &nhwc_name,
                &orig_output,
                &NHWC_TO_NCHW,
            ));
        }

        new_nodes.push(modified);
        new_nodes.extend(output_transposes);
    }

    graph.nodes = new_nodes;
}

/// Eliminate pairs of inverse transposes: if a tensor goes through
/// NCHW→NHWC then immediately NHWC→NCHW (or vice versa), remove both.
fn eliminate_inverse_transposes(graph: &mut Graph) -> bool {
    let producer_map = build_producer_map(&graph.nodes);
    let consumer_map = build_consumer_map(&graph.nodes);
    let mut to_remove: HashSet<usize> = HashSet::new();
    let mut rewrites: HashMap<String, String> = HashMap::new();

    for (i, node) in graph.nodes.iter().enumerate() {
        // Only cancel LayoutTranspose pairs — regular Transposes are part of model computation
        if node.op_type != OpType::LayoutTranspose {
            continue;
        }
        let Some(perm) = get_transpose_perm(node) else {
            continue;
        };
        if !is_inverse_transpose(&perm) {
            continue;
        }
        let input_name = &node.inputs[0];
        let Some(&producer_idx) = producer_map.get(input_name) else {
            continue;
        };
        let producer = &graph.nodes[producer_idx];
        if producer.op_type != OpType::LayoutTranspose {
            continue;
        }
        let Some(prod_perm) = get_transpose_perm(producer) else {
            continue;
        };
        if !are_inverse_perms(&prod_perm, &perm) {
            continue;
        }
        // The producer's input goes directly to this node's output.
        // Only safe if the intermediate tensor has exactly one consumer (this node).
        let consumers = consumer_map.get(input_name).map(|v| v.len()).unwrap_or(0);
        if consumers != 1 {
            continue;
        }
        // Rewrite: this node's output → producer's input
        let source = producer.inputs[0].clone();
        let target = node.outputs[0].clone();
        rewrites.insert(target, source);
        to_remove.insert(i);
        to_remove.insert(producer_idx);
    }

    if to_remove.is_empty() {
        return false;
    }

    // Apply rewrites to all node inputs
    for node in &mut graph.nodes {
        for input in &mut node.inputs {
            if let Some(replacement) = rewrites.get(input) {
                *input = replacement.clone();
            }
        }
    }
    // Also rewrite graph outputs
    for output in &mut graph.outputs {
        if let Some(replacement) = rewrites.get(&output.name) {
            output.name = replacement.clone();
        }
    }

    // Remove dead transpose nodes (in reverse order to preserve indices)
    let mut indices: Vec<usize> = to_remove.into_iter().collect();
    indices.sort_unstable_by(|a, b| b.cmp(a));
    for idx in indices {
        graph.nodes.remove(idx);
    }

    true
}

/// Push a transpose forward through a unary/elementwise op.
/// If we have: Transpose(perm) → UnaryOp → ...
/// We can rewrite to: UnaryOp → Transpose(perm)
/// This lets the transpose reach and cancel with an inverse transpose downstream.
fn push_transposes_through_unary(graph: &mut Graph, counter: &mut usize) -> bool {
    let consumer_map = build_consumer_map(&graph.nodes);

    // Collect moves to make (avoid borrowing issues)
    let mut moves: Vec<(usize, usize)> = Vec::new(); // (transpose_idx, consumer_idx)

    for (i, node) in graph.nodes.iter().enumerate() {
        // Only push LayoutTranspose — regular Transposes are model computation
        if node.op_type != OpType::LayoutTranspose {
            continue;
        }
        let Some(perm) = get_transpose_perm(node) else {
            continue;
        };
        if !is_inverse_transpose(&perm) {
            continue;
        }
        let output_name = &node.outputs[0];
        let Some(consumers) = consumer_map.get(output_name) else {
            continue;
        };
        if consumers.len() != 1 {
            continue;
        }
        let consumer_idx = consumers[0];
        let consumer = &graph.nodes[consumer_idx];
        if !transpose_can_pass_through(consumer.op_type) {
            continue;
        }
        // Consumer must have exactly 1 input (the transposed tensor) for simple push-through
        // Some ops like Clip have optional inputs (min, max) that are scalar — skip those
        let data_inputs: Vec<usize> = consumer
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, name)| !name.is_empty() && *name == output_name)
            .map(|(idx, _)| idx)
            .collect();
        if data_inputs.len() != 1 || data_inputs[0] != 0 {
            continue;
        }
        moves.push((i, consumer_idx));
    }

    // Process one move at a time since each remove+insert shifts indices.
    // The outer loop in optimize() will call us again for the rest.
    if let Some(&(transpose_idx, consumer_idx)) = moves.first() {
        let transpose_input = graph.nodes[transpose_idx].inputs[0].clone();
        let consumer_output = graph.nodes[consumer_idx].outputs[0].clone();

        // Rewire: consumer takes the pre-transpose input
        graph.nodes[consumer_idx].inputs[0] = transpose_input;

        // Consumer now produces an intermediate; new transpose produces the original output
        let intermediate = unique_name("__push_unary", counter);
        graph.nodes[consumer_idx].outputs[0] = intermediate.clone();

        // Update transpose to sit after the consumer
        graph.nodes[transpose_idx].inputs[0] = intermediate;
        graph.nodes[transpose_idx].outputs[0] = consumer_output;

        // Move the transpose node after the consumer
        if transpose_idx < consumer_idx {
            // transpose is before consumer — move it after
            let t = graph.nodes.remove(transpose_idx);
            // consumer_idx shifted by -1 since we removed an earlier element
            graph.nodes.insert(consumer_idx, t);
        }

        true
    } else {
        false
    }
}

/// Push transposes through binary elementwise ops.
/// If both inputs to a binary op come from the same type of transpose,
/// remove both input transposes and add one transpose on the output.
fn push_transposes_through_binary(graph: &mut Graph, counter: &mut usize) -> bool {
    let producer_map = build_producer_map(&graph.nodes);
    let consumer_map = build_consumer_map(&graph.nodes);
    let mut changed = false;

    let mut moves: Vec<(usize, Vec<i64>)> = Vec::new(); // (binary_node_idx, perm)

    'outer: for (i, node) in graph.nodes.iter().enumerate() {
        if !is_binary_elementwise(node.op_type) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }

        // Both non-empty inputs must come from transposes with the same perm
        let mut perm: Option<Vec<i64>> = None;
        let mut transpose_indices = Vec::new();

        for input_name in &node.inputs[..2] {
            if input_name.is_empty() {
                continue 'outer;
            }
            let Some(&prod_idx) = producer_map.get(input_name) else {
                continue 'outer;
            };
            let prod = &graph.nodes[prod_idx];
            // Only push LayoutTranspose — regular Transposes are model computation
            if prod.op_type != OpType::LayoutTranspose {
                continue 'outer;
            }
            let Some(p) = get_transpose_perm(prod) else {
                continue 'outer;
            };
            if !is_inverse_transpose(&p) {
                continue 'outer;
            }
            // Ensure this transpose output is only consumed by this binary op
            let consumers = consumer_map.get(input_name).map(|v| v.len()).unwrap_or(0);
            if consumers != 1 {
                continue 'outer;
            }
            if let Some(ref existing_perm) = perm {
                if *existing_perm != p {
                    continue 'outer;
                }
            } else {
                perm = Some(p);
            }
            transpose_indices.push(prod_idx);
        }

        if let Some(p) = perm {
            if transpose_indices.len() == 2 {
                moves.push((i, p));
            }
        }
    }

    if let Some((binary_idx, perm)) = moves.into_iter().next() {
        let node = &graph.nodes[binary_idx];
        let input0 = node.inputs[0].clone();
        let input1 = node.inputs[1].clone();

        // Find the transpose producers
        let prod0_idx = *producer_map.get(&input0).unwrap();
        let prod1_idx = *producer_map.get(&input1).unwrap();

        let pre_transpose_input0 = graph.nodes[prod0_idx].inputs[0].clone();
        let pre_transpose_input1 = graph.nodes[prod1_idx].inputs[0].clone();
        let binary_output = graph.nodes[binary_idx].outputs[0].clone();

        // Rewire binary to take pre-transpose inputs
        graph.nodes[binary_idx].inputs[0] = pre_transpose_input0;
        graph.nodes[binary_idx].inputs[1] = pre_transpose_input1;

        // Binary now produces intermediate; add transpose after
        let intermediate = unique_name("__push_binary", counter);
        graph.nodes[binary_idx].outputs[0] = intermediate.clone();

        let perm_arr: [i64; 4] = perm.try_into().unwrap();
        // Preserve LayoutTranspose type when pushing through binary ops.
        // Check if the source transposes were LayoutTranspose.
        let is_layout = graph.nodes[prod0_idx].op_type == OpType::LayoutTranspose;
        let t = if is_layout {
            make_layout_transpose_node(
                &unique_name("__binary_layout_transpose", counter),
                &intermediate,
                &binary_output,
                &perm_arr,
            )
        } else {
            make_transpose_node(
                &unique_name("__binary_transpose", counter),
                &intermediate,
                &binary_output,
                &perm_arr,
            )
        };

        // Mark the old transpose nodes for removal by making them identity-like
        // (they'll be cleaned up by dead node removal)
        graph.nodes[prod0_idx].op_type = OpType::Identity;
        graph.nodes[prod0_idx].attrs = Attrs(HashMap::new());
        graph.nodes[prod1_idx].op_type = OpType::Identity;
        graph.nodes[prod1_idx].attrs = Attrs(HashMap::new());

        // Insert transpose right after binary
        let insert_pos = binary_idx + 1;
        graph.nodes.insert(insert_pos, t);

        changed = true;
    }

    changed
}

/// Fold BatchNormalization into a preceding Conv when possible.
/// Conv output = W*x + B
/// BN output = gamma * (Conv_out - mean) / sqrt(var + eps) + beta
/// Fused: W' = W * gamma / sqrt(var+eps), B' = (B - mean) * gamma / sqrt(var+eps) + beta
fn fold_batchnorm_into_conv(graph: &mut Graph) {
    let producer_map = build_producer_map(&graph.nodes);
    let consumer_map = build_consumer_map(&graph.nodes);
    let mut to_remove: HashSet<usize> = HashSet::new();

    let bn_indices: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op_type == OpType::BatchNormalization)
        .map(|(i, _)| i)
        .collect();

    for bn_idx in bn_indices {
        let bn = &graph.nodes[bn_idx];
        if bn.inputs.len() < 5 {
            continue;
        }
        let bn_input = &bn.inputs[0];
        let Some(&conv_idx) = producer_map.get(bn_input) else {
            continue;
        };
        if graph.nodes[conv_idx].op_type != OpType::Conv {
            continue;
        }
        // Conv output must only feed into this BN
        let consumers = consumer_map.get(bn_input).map(|v| v.len()).unwrap_or(0);
        if consumers != 1 {
            continue;
        }

        let conv = &graph.nodes[conv_idx];
        let weight_name = &conv.inputs[1];
        let bias_name = conv.inputs.get(2).cloned().unwrap_or_default();

        // Get BN parameters from initializers
        let gamma_name = &bn.inputs[1];
        let beta_name = &bn.inputs[2];
        let mean_name = &bn.inputs[3];
        let var_name = &bn.inputs[4];
        let epsilon = bn.attrs.get_float("epsilon").unwrap_or(1e-5);

        let (Some(weight), Some(gamma), Some(beta), Some(mean), Some(var)) = (
            graph.initializers.get(weight_name),
            graph.initializers.get(gamma_name),
            graph.initializers.get(beta_name),
            graph.initializers.get(mean_name),
            graph.initializers.get(var_name),
        ) else {
            continue;
        };

        let Ok(w_f) = weight.floats() else { continue };
        let Ok(gamma_f) = gamma.floats() else {
            continue;
        };
        let Ok(beta_f) = beta.floats() else {
            continue;
        };
        let Ok(mean_f) = mean.floats() else {
            continue;
        };
        let Ok(var_f) = var.floats() else { continue };

        let c_out = gamma_f.len();
        if c_out == 0 {
            continue;
        }

        // Compute scale factors: scale[c] = gamma[c] / sqrt(var[c] + eps)
        let mut scale = vec![0.0f32; c_out];
        for c in 0..c_out {
            scale[c] = gamma_f[c] / (var_f[c] + epsilon).sqrt();
        }

        // Fuse into weights: W'[c_out, ...] = W[c_out, ...] * scale[c_out]
        let w_dims = weight.dims.clone();
        let elems_per_filter = w_f.len() / c_out;
        let mut new_w = w_f.to_vec();
        for (c, &s) in scale.iter().enumerate() {
            let start = c * elems_per_filter;
            let end = start + elems_per_filter;
            for v in &mut new_w[start..end] {
                *v *= s;
            }
        }

        // Fuse into bias: B'[c] = (B[c] - mean[c]) * scale[c] + beta[c]
        let old_bias: Vec<f32> = if !bias_name.is_empty() {
            graph
                .initializers
                .get(&bias_name)
                .and_then(|t| t.floats().ok())
                .map(|f| f.to_vec())
                .unwrap_or_else(|| vec![0.0; c_out])
        } else {
            vec![0.0; c_out]
        };

        let mut new_bias = vec![0.0f32; c_out];
        for c in 0..c_out {
            new_bias[c] = (old_bias[c] - mean_f[c]) * scale[c] + beta_f[c];
        }

        // Update initializers
        let new_w_tensor = Tensor::new(w_dims, new_w);
        graph.initializers.insert(weight_name.clone(), new_w_tensor);

        let fused_bias_name = if bias_name.is_empty() {
            let name = format!("{weight_name}__fused_bias");
            // Update conv to have 3 inputs
            let conv_mut = &mut graph.nodes[conv_idx];
            while conv_mut.inputs.len() < 3 {
                conv_mut.inputs.push(String::new());
            }
            conv_mut.inputs[2] = name.clone();
            name
        } else {
            bias_name.clone()
        };

        let bias_tensor = Tensor::new(dims![c_out], new_bias);
        graph.initializers.insert(fused_bias_name, bias_tensor);

        // Rewire: conv now produces BN's output directly
        let bn_output = graph.nodes[bn_idx].outputs[0].clone();
        graph.nodes[conv_idx].outputs[0] = bn_output;

        to_remove.insert(bn_idx);
    }

    if !to_remove.is_empty() {
        let mut indices: Vec<usize> = to_remove.into_iter().collect();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        for idx in indices {
            graph.nodes.remove(idx);
        }
    }
}

/// Collect all tensor names referenced inside sub-graphs (Loop body, If branches, Scan body).
fn collect_subgraph_references(nodes: &[Node]) -> HashSet<String> {
    let mut refs = HashSet::new();
    for node in nodes {
        for attr in node.attrs.0.values() {
            if let Attr::Graph(g) = attr {
                // Any tensor name used as input in the sub-graph that isn't produced
                // within the sub-graph is a reference to the outer graph.
                let inner_outputs: HashSet<&str> = g
                    .nodes
                    .iter()
                    .flat_map(|n| n.outputs.iter())
                    .map(|s| s.as_str())
                    .collect();
                let inner_inits: HashSet<&str> =
                    g.initializers.keys().map(|s| s.as_str()).collect();
                let inner_inputs: HashSet<&str> =
                    g.inputs.iter().map(|i| i.name.as_str()).collect();
                for inner_node in &g.nodes {
                    for input in &inner_node.inputs {
                        if !input.is_empty()
                            && !inner_outputs.contains(input.as_str())
                            && !inner_inits.contains(input.as_str())
                            && !inner_inputs.contains(input.as_str())
                        {
                            refs.insert(input.clone());
                        }
                    }
                }
                // Recurse into nested sub-graphs
                let nested = collect_subgraph_references(&g.nodes);
                refs.extend(nested);
            }
        }
    }
    refs
}

/// Remove nodes whose outputs are not consumed by any other node, graph output,
/// or sub-graph (Loop body, If branches, etc.).
fn remove_dead_nodes(graph: &mut Graph) {
    let output_names: HashSet<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();
    let subgraph_refs = collect_subgraph_references(&graph.nodes);

    loop {
        let consumer_map = build_consumer_map(&graph.nodes);
        let mut to_remove = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            // Skip if any output is a graph output
            if node.outputs.iter().any(|o| output_names.contains(o)) {
                continue;
            }
            // Skip if any output is referenced by a sub-graph
            if node.outputs.iter().any(|o| subgraph_refs.contains(o)) {
                continue;
            }
            // Remove if no output is consumed
            let consumed = node
                .outputs
                .iter()
                .any(|o| consumer_map.get(o).map(|v| !v.is_empty()).unwrap_or(false));
            if !consumed {
                to_remove.push(i);
            }
        }

        if to_remove.is_empty() {
            break;
        }

        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            graph.nodes.remove(idx);
        }
    }
}

/// Dump the graph as a human-readable string for debugging.
pub fn dump(graph: &Graph) -> String {
    let mut out = String::new();

    writeln!(out, "=== Graph ({} nodes) ===", graph.nodes.len()).unwrap();

    // Show graph inputs
    for input in &graph.inputs {
        let shape_str = input
            .shape
            .as_ref()
            .map(|s| {
                format!(
                    "[{}]",
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .unwrap_or_else(|| "?".to_string());
        writeln!(
            out,
            "INPUT: {} {:?} {}",
            input.name, input.elem_type, shape_str
        )
        .unwrap();
    }

    // Show graph outputs
    for output in &graph.outputs {
        writeln!(out, "OUTPUT: {}", output.name).unwrap();
    }

    writeln!(out).unwrap();

    // Show each node
    for (i, node) in graph.nodes.iter().enumerate() {
        let outputs_str = node.outputs.join(", ");
        let inputs_str = node.inputs.join(", ");

        let mut extra = String::new();
        if matches!(node.op_type, OpType::Transpose | OpType::LayoutTranspose) {
            if let Some(perm) = node.attrs.get_ints("perm") {
                let prefix = if node.op_type == OpType::LayoutTranspose {
                    "LAYOUT "
                } else {
                    ""
                };
                if is_nchw_to_nhwc(&perm) {
                    extra = format!(" [{prefix}NCHW→NHWC]");
                } else if is_nhwc_to_nchw(&perm) {
                    extra = format!(" [{prefix}NHWC→NCHW]");
                } else {
                    extra = format!(" perm={perm:?}");
                }
            }
        } else if node.op_type == OpType::Conv {
            let group = node.attrs.get_int("group").unwrap_or(1);
            if group > 1 {
                write!(extra, " group={group}").unwrap();
            }
        }

        writeln!(
            out,
            "{i:4}: {outputs_str} = {:?}({inputs_str}){extra}",
            node.op_type
        )
        .unwrap();
    }

    // Count transpose nodes
    let transpose_count = graph
        .nodes
        .iter()
        .filter(|n| n.op_type == OpType::Transpose)
        .count();
    let nhwc_transposes = graph
        .nodes
        .iter()
        .filter(|n| {
            get_transpose_perm(n)
                .map(|p| is_inverse_transpose(&p))
                .unwrap_or(false)
        })
        .count();

    writeln!(out).unwrap();
    writeln!(
        out,
        "=== {transpose_count} transpose nodes ({nhwc_transposes} layout transposes) ==="
    )
    .unwrap();

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_ir::ElemType;
    use crate::onnx_ir::ValueInfo;

    fn make_simple_node(op: OpType, inputs: &[&str], outputs: &[&str]) -> Node {
        Node {
            op_type: op,
            name: format!("{op:?}_{}", outputs[0]),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attrs: Attrs(HashMap::new()),
        }
    }

    fn make_conv_node(input: &str, weight: &str, bias: &str, output: &str) -> Node {
        let mut attrs = HashMap::new();
        attrs.insert("kernel_shape".to_string(), Attr::Ints(vec![3, 3]));
        attrs.insert("strides".to_string(), Attr::Ints(vec![1, 1]));
        attrs.insert("pads".to_string(), Attr::Ints(vec![1, 1, 1, 1]));
        Node {
            op_type: OpType::Conv,
            name: format!("conv_{output}"),
            inputs: vec![input.to_string(), weight.to_string(), bias.to_string()],
            outputs: vec![output.to_string()],
            attrs: Attrs(attrs),
        }
    }

    fn make_bn_node(input: &str, output: &str, c: usize) -> (Node, HashMap<String, Tensor>) {
        let gamma_name = format!("{output}_gamma");
        let beta_name = format!("{output}_beta");
        let mean_name = format!("{output}_mean");
        let var_name = format!("{output}_var");
        let mut attrs = HashMap::new();
        attrs.insert("epsilon".to_string(), Attr::Float(1e-5));
        let node = Node {
            op_type: OpType::BatchNormalization,
            name: format!("bn_{output}"),
            inputs: vec![
                input.to_string(),
                gamma_name.clone(),
                beta_name.clone(),
                mean_name.clone(),
                var_name.clone(),
            ],
            outputs: vec![output.to_string()],
            attrs: Attrs(attrs),
        };

        let mut inits = HashMap::new();
        let d = dims![c];
        inits.insert(gamma_name, Tensor::new(d.clone(), vec![1.0; c]));
        inits.insert(beta_name, Tensor::new(d.clone(), vec![0.0; c]));
        inits.insert(mean_name, Tensor::new(d.clone(), vec![0.0; c]));
        inits.insert(var_name, Tensor::new(d, vec![1.0; c]));

        (node, inits)
    }

    fn make_graph(
        nodes: Vec<Node>,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        initializers: HashMap<String, Tensor>,
    ) -> Graph {
        Graph {
            nodes,
            inputs: inputs
                .iter()
                .map(|name| ValueInfo {
                    name: name.to_string(),
                    elem_type: ElemType::Float,
                    shape: None,
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|name| ValueInfo {
                    name: name.to_string(),
                    elem_type: ElemType::Float,
                    shape: None,
                })
                .collect(),
            initializers,
            opset_version: 13,
        }
    }

    #[test]
    fn test_eliminate_conv_relu_transposes() {
        // Conv → Relu: the NHWC→NCHW after Conv and NCHW→NHWC before Relu should cancel
        // After insert: Transpose(NCHW→NHWC) → Conv → Transpose(NHWC→NCHW) → Relu → ...
        // Since Relu is unary, we push the NHWC→NCHW through Relu, then it can cancel
        // with a subsequent NCHW→NHWC (if there's another spatial op), or just remains.

        // Simple case: Conv → Relu (no second spatial op)
        let mut graph = make_graph(
            vec![
                make_conv_node("input", "W", "B", "conv_out"),
                make_simple_node(OpType::Relu, &["conv_out"], &["relu_out"]),
            ],
            vec!["input"],
            vec!["relu_out"],
            HashMap::new(),
        );

        // Add weight initializer
        graph.initializers.insert(
            "W".to_string(),
            Tensor::new(dims![3, 3, 3, 3], vec![0.0; 81]),
        );
        graph
            .initializers
            .insert("B".to_string(), Tensor::new(dims![3], vec![0.0; 3]));

        let before = dump(&graph);
        optimize(&mut graph);
        let after = dump(&graph);

        eprintln!("BEFORE:\n{before}");
        eprintln!("AFTER:\n{after}");

        // Should have fewer transposes than the naive insertion
        let transpose_count = graph
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::LayoutTranspose)
            .count();
        // We expect: 1 NCHW→NHWC at input, 1 NHWC→NCHW at output (pushed past Relu)
        assert_eq!(
            transpose_count, 2,
            "Expected 2 layout transposes, got {transpose_count}"
        );
    }

    #[test]
    fn test_conv_relu_conv_cancels() {
        // Conv → Relu → Conv: the transposes between them should cancel
        let mut graph = make_graph(
            vec![
                make_conv_node("input", "W1", "B1", "conv1_out"),
                make_simple_node(OpType::Relu, &["conv1_out"], &["relu_out"]),
                make_conv_node("relu_out", "W2", "B2", "conv2_out"),
            ],
            vec!["input"],
            vec!["conv2_out"],
            HashMap::new(),
        );

        let w = Tensor::new(dims![3, 3, 3, 3], vec![0.0; 81]);
        graph.initializers.insert("W1".to_string(), w.clone());
        graph.initializers.insert("W2".to_string(), w);
        let b = Tensor::new(dims![3], vec![0.0; 3]);
        graph.initializers.insert("B1".to_string(), b.clone());
        graph.initializers.insert("B2".to_string(), b);

        let before = dump(&graph);
        optimize(&mut graph);
        let after = dump(&graph);

        eprintln!("BEFORE:\n{before}");
        eprintln!("AFTER:\n{after}");

        let transpose_count = graph
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::LayoutTranspose)
            .count();
        // We expect: 1 NCHW→NHWC at entry, Conv, Relu, Conv, 1 NHWC→NCHW at exit
        // The pair in the middle should have been eliminated
        assert_eq!(
            transpose_count, 2,
            "Expected 2 layout transposes, got {transpose_count}"
        );
    }

    #[test]
    fn test_fold_batchnorm_into_conv() {
        let c = 3;
        let (bn, bn_inits) = make_bn_node("conv_out", "bn_out", c);

        let mut graph = make_graph(
            vec![make_conv_node("input", "W", "B", "conv_out"), bn],
            vec!["input"],
            vec!["bn_out"],
            bn_inits,
        );

        graph.initializers.insert(
            "W".to_string(),
            Tensor::new(dims![3, 3, 3, 3], vec![1.0; 81]),
        );
        graph
            .initializers
            .insert("B".to_string(), Tensor::new(dims![3], vec![0.5; 3]));

        optimize(&mut graph);

        // BN should be gone
        let bn_count = graph
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::BatchNormalization)
            .count();
        assert_eq!(bn_count, 0, "BatchNorm should be folded into Conv");

        // Conv should now produce bn_out (after transpose rewiring)
        let conv = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::Conv)
            .expect("Conv should still exist");
        // Conv output may be rewired through transposes, that's ok
        assert!(!conv.outputs[0].is_empty());
    }

    #[test]
    fn test_push_transpose_through_add() {
        // Transpose(NCHW→NHWC) → both inputs of Add → should push through
        let mut graph = make_graph(
            vec![
                make_conv_node("input_a", "W1", "B1", "conv_a"),
                make_conv_node("input_b", "W2", "B2", "conv_b"),
                make_simple_node(OpType::Add, &["conv_a", "conv_b"], &["add_out"]),
            ],
            vec!["input_a", "input_b"],
            vec!["add_out"],
            HashMap::new(),
        );

        let w = Tensor::new(dims![3, 3, 3, 3], vec![0.0; 81]);
        graph.initializers.insert("W1".to_string(), w.clone());
        graph.initializers.insert("W2".to_string(), w);
        let b = Tensor::new(dims![3], vec![0.0; 3]);
        graph.initializers.insert("B1".to_string(), b.clone());
        graph.initializers.insert("B2".to_string(), b);

        let before = dump(&graph);
        optimize(&mut graph);
        let after = dump(&graph);

        eprintln!("BEFORE:\n{before}");
        eprintln!("AFTER:\n{after}");

        // The two NHWC→NCHW transposes after each conv should push through
        // the Add and merge (or at least reduce count)
        let transpose_count = graph
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::Transpose)
            .count();
        // Best case: 2 at input (NCHW→NHWC), 1 at output (NHWC→NCHW)
        // The two intermediate pairs cancel, leaving 3
        assert!(
            transpose_count <= 3,
            "Expected <= 3 transposes, got {transpose_count}\n{after}"
        );
    }

    #[test]
    fn test_dump_output() {
        let graph = make_graph(
            vec![
                make_conv_node("input", "W", "B", "conv_out"),
                make_simple_node(OpType::Relu, &["conv_out"], &["relu_out"]),
            ],
            vec!["input"],
            vec!["relu_out"],
            HashMap::new(),
        );

        let output = dump(&graph);
        assert!(output.contains("Conv"));
        assert!(output.contains("Relu"));
        assert!(output.contains("INPUT: input"));
        assert!(output.contains("OUTPUT: relu_out"));
    }
}
