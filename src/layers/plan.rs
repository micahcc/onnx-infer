use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Dims;
use crate::Layout;
use crate::Result;
use crate::ShapeLayout;
use crate::Tensor;
use crate::dims;
use crate::layers::Layer;
use crate::layers::OpType;
use crate::layers::abs;
use crate::layers::add;
use crate::layers::and;
use crate::layers::argmax;
use crate::layers::auto_cast;
use crate::layers::average_pool;
use crate::layers::batch_norm;
use crate::layers::cast;
use crate::layers::category_mapper;
use crate::layers::ceil;
use crate::layers::clip;
use crate::layers::compress;
use crate::layers::concat;
use crate::layers::constant;
use crate::layers::constant_of_shape;
use crate::layers::conv;
use crate::layers::dequantize_linear;
use crate::layers::div;
use crate::layers::dropout;
use crate::layers::equal;
use crate::layers::exp;
use crate::layers::expand;
use crate::layers::flatten;
use crate::layers::floor;
use crate::layers::gather;
use crate::layers::gemm;
use crate::layers::global_avg_pool;
use crate::layers::greater;
use crate::layers::hardmax;
use crate::layers::identity;
use crate::layers::if_op;
use crate::layers::leaky_relu;
use crate::layers::less;
use crate::layers::log;
use crate::layers::loop_op;
use crate::layers::lrn;
use crate::layers::lstm;
use crate::layers::matmul;
use crate::layers::max_op;
use crate::layers::maxpool;
use crate::layers::min_op;
use crate::layers::mul;
use crate::layers::nms;
use crate::layers::nonzero;
use crate::layers::not;
use crate::layers::prelu;
use crate::layers::qlinear_add;
use crate::layers::qlinear_conv;
use crate::layers::qlinear_global_avg_pool;
use crate::layers::qlinear_matmul;
use crate::layers::quantize_linear;
use crate::layers::range;
use crate::layers::reduce_max;
use crate::layers::reduce_mean;
use crate::layers::reduce_min;
use crate::layers::reduce_sum;
use crate::layers::relu;
use crate::layers::reshape;
use crate::layers::resize;
use crate::layers::roi_align;
use crate::layers::round;
use crate::layers::scan;
use crate::layers::scatter_elements;
use crate::layers::shape_op;
use crate::layers::sigmoid;
use crate::layers::slice;
use crate::layers::softmax;
use crate::layers::softplus;
use crate::layers::split;
use crate::layers::sqrt;
use crate::layers::squeeze;
use crate::layers::sub;
use crate::layers::sum;
use crate::layers::tanh;
use crate::layers::tile;
use crate::layers::topk;
use crate::layers::transpose;
use crate::layers::unary_ops;
use crate::layers::unsqueeze;
use crate::layers::where_op;
use crate::onnx_ir::Attr;
use crate::onnx_ir::Graph;
use crate::onnx_ir::Node;

pub enum PlanNode {
    Single {
        output: String,
        layer: Box<dyn Layer>,
    },
    Loop(Box<loop_op::Loop>),
    Split(Box<split::Split>),
    If(Box<if_op::If>),
    TopK(Box<topk::TopK>),
    Scan(Box<scan::Scan>),
    #[cfg(feature = "xnnpack")]
    XnnpackSubgraph(Box<super::xnnpack_subgraph::XnnpackSubgraph>),
}

pub struct Plan {
    pub nodes: Vec<PlanNode>,
    pub initializers: HashMap<String, Tensor>,
    pub output_names: Vec<String>,
    pub shape_map: HashMap<String, ShapeLayout>,
    pub type_map: HashMap<String, DType>,
    pub tensor_pool: HashMap<String, Tensor>,
}

impl Plan {
    pub fn build(graph: &Graph, input_sizes: &HashMap<String, Dims>) -> Result<Self> {
        Self::build_full(graph, input_sizes, &HashMap::new(), &HashMap::new())
    }

    pub fn build_with_types(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
    ) -> Result<Self> {
        Self::build_full(graph, input_sizes, type_hints, &HashMap::new())
    }

    #[cfg(feature = "xnnpack")]
    pub fn build_with_xnnpack(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
    ) -> Result<Self> {
        Self::build_full_inner(graph, input_sizes, &HashMap::new(), &HashMap::new(), true)
    }

    pub fn build_full(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
        input_values: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::build_full_inner(graph, input_sizes, type_hints, input_values, false)
    }

    fn build_full_inner(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
        input_values: &HashMap<String, Tensor>,
        #[allow(unused)] enable_xnnpack: bool,
    ) -> Result<Self> {
        let mut initializers = graph.initializers.clone();

        let output_names: Vec<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();

        let mut type_map: HashMap<String, DType> = HashMap::new();
        for (name, tensor) in &initializers {
            type_map.insert(name.clone(), tensor.dtype());
        }
        for (name, &dtype) in type_hints {
            type_map.insert(name.clone(), dtype);
        }
        for input in &graph.inputs {
            if !type_map.contains_key(&input.name) {
                let dtype = input.elem_type.to_dtype();
                type_map.insert(input.name.clone(), dtype);
            }
        }

        let mut shape_map: HashMap<String, ShapeLayout> = HashMap::new();
        for (name, tensor) in &initializers {
            shape_map.insert(
                name.clone(),
                ShapeLayout::new(tensor.dims.clone(), tensor.layout),
            );
        }
        let initializer_names: std::collections::HashSet<&str> =
            initializers.keys().map(|k| k.as_str()).collect();
        for input in &graph.inputs {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = &input.shape {
                if shape.iter().all(|&d| d > 0) || input_sizes.contains_key(&input.name) {
                    shape_map.insert(input.name.clone(), ShapeLayout::nchw(shape.clone()));
                }
            }
        }
        for (name, user_dims) in input_sizes {
            if let Some(existing) = shape_map.get_mut(name) {
                if existing.dims.len() == user_dims.len() {
                    for (i, d) in existing.dims.iter_mut().enumerate() {
                        if *d == 0 {
                            *d = user_dims[i];
                        }
                    }
                } else {
                    existing.dims = user_dims.clone();
                }
            } else {
                shape_map.insert(name.clone(), ShapeLayout::nchw(user_dims.clone()));
            }
        }

        let mut known_values: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in input_values {
            known_values.insert(name.clone(), tensor.clone());
        }


        let mut nodes = Vec::new();
        #[cfg(feature = "xnnpack")]
        let mut node_meta: Vec<Option<(OpType, Vec<String>, Node)>> = Vec::new();
        let mut cast_counter = 0usize;
        for node in &graph.nodes {
            let op = node.op_type;

            let expected = op.expected_input_dtypes();
            let mut modified_inputs = node.inputs.clone();

            let mut input_types = Vec::new();
            for name in &node.inputs {
                if let Some(&dt) = type_map.get(name) {
                    input_types.push(dt);
                }
            }

            for (i, input_name) in node.inputs.iter().enumerate() {
                if input_name.is_empty() {
                    continue;
                }
                if let Some(Some(expected_dt)) = expected.get(i) {
                    if let Some(&actual_dt) = type_map.get(input_name) {
                        if actual_dt != *expected_dt {
                            let cast_name = format!("__auto_cast_{cast_counter}__");
                            cast_counter += 1;
                            nodes.push(PlanNode::Single {
                                output: cast_name.clone(),
                                layer: Box::new(auto_cast::AutoCastF32::new(input_name.clone())),
                            });
                            #[cfg(feature = "xnnpack")]
                            node_meta.push(None);
                            type_map.insert(cast_name.clone(), DType::Float);
                            modified_inputs[i] = cast_name;
                        }
                    }
                }
            }

            let out_dtype = op.infer_output_dtype(node, &input_types);
            let out_name = node.outputs.first().filter(|s| !s.is_empty());
            if let Some(out_name) = out_name {
                type_map.insert(out_name.clone(), out_dtype);
            }

            if let Some(mut tensor) = try_propagate_value(
                op,
                node,
                &node.inputs,
                &known_values,
                &initializers,
                &shape_map,
            ) {
                if let Some(out_name) = out_name {
                    // Set layout based on op type (constant-folded tensors default to NCHW)
                    tensor.layout = infer_output_layout(op, node, &shape_map);
                    shape_map.insert(
                        out_name.clone(),
                        ShapeLayout::new(tensor.dims.clone(), tensor.layout),
                    );
                    initializers.insert(out_name.clone(), tensor.clone());
                    known_values.insert(out_name.clone(), tensor);
                }
                // Skip adding to plan — output is already in initializers
                // and will be available at runtime via the values map.
                continue;
            }
            if let Some(shape) =
                op.infer_output_shape(node, &node.inputs, &shape_map, &known_values)
            {
                if let Some(out_name) = out_name {
                    let out_layout = infer_output_layout(op, node, &shape_map);
                    shape_map.insert(
                        out_name.clone(),
                        ShapeLayout::new(shape, out_layout),
                    );
                }
            } else if matches!(op, OpType::Identity | OpType::Transpose | OpType::Conv) {
                if let Some(out_name) = out_name {
                    let missing: Vec<&str> = node
                        .inputs
                        .iter()
                        .filter(|n| !n.is_empty() && !shape_map.contains_key(n.as_str()))
                        .map(|n| n.as_str())
                        .collect();
                    if !missing.is_empty() {
                        eprintln!(
                            "  shape-miss: {:?} {:?} -> {:?}, missing inputs: {:?}",
                            op, node.name, out_name, missing
                        );
                    }
                }
            }

            // For Split, infer types and shapes for all outputs
            if op == OpType::Split {
                let in_dtype = input_types.first().copied().unwrap_or(DType::Float);
                for out_name in &node.outputs {
                    if !out_name.is_empty() {
                        type_map.insert(out_name.clone(), in_dtype);
                    }
                }
                if let Some(in_sl) = node
                    .inputs
                    .first()
                    .filter(|s| !s.is_empty())
                    .and_then(|n| shape_map.get(n))
                    .cloned()
                {
                    let axis_attr = node.attrs.get_int("axis").unwrap_or(0);
                    let rank = in_sl.dims.len() as i64;
                    let axis = if axis_attr < 0 {
                        (rank + axis_attr) as usize
                    } else {
                        axis_attr as usize
                    };
                    let split_sizes = node.attrs.get_ints("split");
                    let num_outputs = node.outputs.len();
                    for (i, out_name) in node.outputs.iter().enumerate() {
                        if out_name.is_empty() {
                            continue;
                        }
                        let mut out_shape = in_sl.dims.clone();
                        out_shape[axis] = if let Some(ref sizes) = split_sizes {
                            sizes[i] as usize
                        } else {
                            let base = in_sl.dims[axis] / num_outputs;
                            let rem = in_sl.dims[axis] % num_outputs;
                            base + if i < rem { 1 } else { 0 }
                        };
                        shape_map.insert(
                            out_name.clone(),
                            ShapeLayout::new(out_shape, in_sl.layout),
                        );
                    }
                }
            }

            // Try to fold Loop ops when all inputs are known
            if op == OpType::Loop {
                let all_inputs_known = node.inputs.iter().all(|n| {
                    n.is_empty() || known_values.contains_key(n) || initializers.contains_key(n)
                });

                if all_inputs_known {
                    let plan_node = build_node_with_opset(
                        op,
                        node,
                        modified_inputs.clone(),
                        &shape_map,
                        graph.opset_version,
                    )?;
                    if let PlanNode::Loop(mut loop_layer) = plan_node {
                        let mut temp_values: HashMap<String, Tensor> = HashMap::new();
                        for name in &node.inputs {
                            if !name.is_empty() {
                                if let Some(t) =
                                    known_values.get(name).or_else(|| initializers.get(name))
                                {
                                    temp_values.insert(name.clone(), t.clone());
                                }
                            }
                        }
                        if let Ok(()) = loop_layer.execute(&mut temp_values) {
                            let mut folded = true;
                            for out_name in &node.outputs {
                                if !out_name.is_empty() {
                                    if let Some(t) = temp_values.remove(out_name) {
                                        shape_map.insert(
                                            out_name.clone(),
                                            ShapeLayout::new(t.dims.clone(), t.layout),
                                        );
                                        initializers.insert(out_name.clone(), t.clone());
                                        known_values.insert(out_name.clone(), t);
                                    } else {
                                        folded = false;
                                    }
                                }
                            }
                            if folded {
                                continue;
                            }
                        }
                    }
                }
            }

            #[cfg(feature = "xnnpack")]
            node_meta.push(Some((op, modified_inputs.clone(), node.clone())));
            nodes.push(build_node_with_opset(
                op,
                node,
                modified_inputs,
                &shape_map,
                graph.opset_version,
            )?);
        }

        // XNNPACK subgraph compilation
        #[cfg(feature = "xnnpack")]
        if enable_xnnpack {
            nodes = compile_xnnpack_subgraphs(
                nodes,
                node_meta,
                &mut shape_map,
                &type_map,
                &initializers,
                graph.opset_version,
                &output_names,
            )?;
        }

        // Pre-allocate tensors for all known shapes/types
        let mut tensor_pool: HashMap<String, Tensor> = HashMap::new();
        // Cap pre-allocation at 256MB per tensor to avoid capacity overflow
        // from shape inference mismatches (e.g., NHWC shapes computed with NCHW logic)
        const MAX_PREALLOC_ELEMS: usize = 256 * 1024 * 1024 / 4;
        for (name, sl) in &shape_map {
            if initializers.contains_key(name) {
                continue;
            }
            let numel: usize = sl.dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).unwrap_or(0);
            if numel == 0 || numel > MAX_PREALLOC_ELEMS {
                continue;
            }
            let dtype = type_map.get(name).copied().unwrap_or(DType::Float);
            let mut tensor = match dtype {
                DType::Float => Tensor::new(sl.dims.clone(), vec![0.0; numel]),
                DType::Int64 => Tensor::new_i64(sl.dims.clone(), vec![0; numel]),
                DType::String => Tensor::new_strings(sl.dims.clone(), vec![vec![]; numel]),
            };
            tensor.layout = sl.layout;
            tensor_pool.insert(name.clone(), tensor);
        }

        // Log plan summary
        {
            let mut cpu_ops = 0usize;
            #[allow(unused_mut)]
            let mut xnnpack_ops = 0usize;
            for node in &nodes {
                match node {
                    PlanNode::Single { .. } => cpu_ops += 1,
                    PlanNode::Loop(_)
                    | PlanNode::Split(_)
                    | PlanNode::If(_)
                    | PlanNode::TopK(_)
                    | PlanNode::Scan(_) => cpu_ops += 1,
                    #[cfg(feature = "xnnpack")]
                    PlanNode::XnnpackSubgraph(sg) => {
                        xnnpack_ops += sg.ops.len();
                    }
                }
            }
            #[cfg(feature = "xnnpack")]
            {
                let total = cpu_ops + xnnpack_ops;
                if total > 0 {
                    let pct = (xnnpack_ops as f64 / total as f64 * 100.0) as u32;
                    tracing::info!("plan: {total} ops, {xnnpack_ops} XNNPACK ({pct}%), {cpu_ops} CPU");
                }
            }
            #[cfg(not(feature = "xnnpack"))]
            {
                let _ = xnnpack_ops;
                let folded = graph.nodes.len().saturating_sub(cpu_ops);
                tracing::info!("plan: {cpu_ops} ops (CPU), {folded} folded");
            }
        }

        Ok(Self {
            nodes,
            initializers,
            output_names,
            shape_map,
            type_map,
            tensor_pool,
        })
    }
}

fn try_propagate_value(
    op: OpType,
    node: &Node,
    input_names: &[String],
    known_values: &HashMap<String, Tensor>,
    initializers: &HashMap<String, Tensor>,
    shape_map: &HashMap<String, ShapeLayout>,
) -> Option<Tensor> {
    if op == OpType::Shape {
        let name = input_names.first().filter(|s| !s.is_empty())?;
        let sl = shape_map.get(name)?;
        if sl.dims.contains(&0) {
            return None;
        }
        let dims: Vec<i64> = sl.dims.iter().map(|&d| d as i64).collect();
        return Some(Tensor::new_i64(dims![dims.len()], dims));
    }

    if op == OpType::Constant {
        return match node.attrs.get("value") {
            Some(Attr::Tensor(t)) => Some(t.clone()),
            _ => None,
        };
    }

    // Skip multi-output and control-flow ops — those are handled separately
    match op {
        OpType::Loop | OpType::Split | OpType::If | OpType::TopK | OpType::Scan => {
            return None;
        }
        _ => {}
    }

    let mut temp_values = HashMap::new();
    for name in input_names {
        if name.is_empty() {
            continue;
        }
        if let Some(t) = known_values.get(name).or_else(|| initializers.get(name)) {
            temp_values.insert(name.clone(), t.clone());
        } else {
            return None;
        }
    }

    let plan_node = build_node(op, node, input_names.to_vec(), shape_map).ok()?;
    if let PlanNode::Single { mut layer, .. } = plan_node {
        let mut output = Tensor::default();
        layer.execute(&temp_values, &mut output).ok()?;
        Some(output)
    } else {
        None
    }
}

pub fn build_node(
    op: OpType,
    node: &Node,
    inputs: Vec<String>,
    shape_map: &HashMap<String, ShapeLayout>,
) -> Result<PlanNode> {
    build_node_with_opset(op, node, inputs, shape_map, 0)
}

pub fn build_node_with_opset(
    op: OpType,
    node: &Node,
    inputs: Vec<String>,
    shape_map: &HashMap<String, ShapeLayout>,
    opset_version: i64,
) -> Result<PlanNode> {
    if op == OpType::Loop {
        let body = match node.attrs.get("body") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("Loop: no body graph"),
        };
        return Ok(PlanNode::Loop(Box::new(loop_op::Loop::new(
            inputs,
            node.outputs.clone(),
            body,
        ))));
    }

    if op == OpType::Split {
        let axis = node.attrs.get_int("axis").unwrap_or(0);
        let split_sizes = node.attrs.get_ints("split").unwrap_or_default();
        return Ok(PlanNode::Split(Box::new(split::Split::new(
            inputs,
            node.outputs.clone(),
            axis,
            split_sizes,
        ))));
    }

    if op == OpType::If {
        let then_branch = match node.attrs.get("then_branch") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("If: no then_branch"),
        };
        let else_branch = match node.attrs.get("else_branch") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("If: no else_branch"),
        };
        return Ok(PlanNode::If(Box::new(if_op::If::new(
            inputs,
            node.outputs.clone(),
            then_branch,
            else_branch,
        ))));
    }

    if op == OpType::TopK {
        let axis = node.attrs.get_int("axis").unwrap_or(-1);
        let largest = node.attrs.get_int("largest").unwrap_or(1) != 0;
        return Ok(PlanNode::TopK(Box::new(topk::TopK::new(
            inputs,
            node.outputs.clone(),
            axis,
            largest,
        ))));
    }

    if op == OpType::Scan {
        let body = match node.attrs.get("body") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("Scan: no body graph"),
        };
        let num_scan_inputs = node.attrs.get_int("num_scan_inputs").unwrap_or(0) as usize;
        let scan_input_directions = node
            .attrs
            .get_ints("scan_input_directions")
            .unwrap_or_default();
        let scan_output_directions = node
            .attrs
            .get_ints("scan_output_directions")
            .unwrap_or_default();
        return Ok(PlanNode::Scan(Box::new(scan::Scan::new(
            inputs,
            node.outputs.clone(),
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
        ))));
    }

    let output = if node.outputs.is_empty() || node.outputs[0].is_empty() {
        String::new()
    } else {
        node.outputs[0].clone()
    };

    // Pre-resolve input shapes before moving inputs into constructors
    let empty: &[usize] = &[];
    let mut input_shapes: [&[usize]; 8] = [empty; 8];
    for (i, name) in inputs.iter().enumerate().take(8) {
        if !name.is_empty() {
            if let Some(sl) = shape_map.get(name) {
                input_shapes[i] = sl.dims.as_slice();
            }
        }
    }

    let layer: Box<dyn Layer> = match op {
        OpType::Relu => Box::new(relu::Relu::new(inputs)),
        OpType::LeakyRelu => Box::new(leaky_relu::LeakyRelu::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(0.01),
        )),
        OpType::Clip => Box::new(clip::Clip::new(
            inputs,
            node.attrs.get_float("min").unwrap_or(f32::NEG_INFINITY),
            node.attrs.get_float("max").unwrap_or(f32::INFINITY),
        )),
        OpType::BatchNormalization => Box::new(batch_norm::BatchNorm::new(
            inputs,
            node.attrs.get_float("epsilon").unwrap_or(1e-5),
            input_shapes[0],
        )),
        OpType::Sigmoid => Box::new(sigmoid::Sigmoid::new(inputs)),
        OpType::Exp => Box::new(exp::Exp::new(inputs)),
        OpType::Log => Box::new(log::Log::new(inputs)),
        OpType::Lrn => Box::new(lrn::Lrn::new(
            inputs,
            node.attrs.get_int("size").unwrap_or(1) as usize,
            node.attrs.get_float("alpha").unwrap_or(0.0001),
            node.attrs.get_float("beta").unwrap_or(0.75),
            node.attrs.get_float("bias").unwrap_or(1.0),
        )),
        OpType::Tanh => Box::new(tanh::Tanh::new(inputs)),
        OpType::Expand => Box::new(expand::Expand::new(inputs)),
        OpType::Less => Box::new(less::Less::new(inputs)),
        OpType::Equal => Box::new(equal::Equal::new(inputs)),
        OpType::Greater => Box::new(greater::Greater::new(inputs)),
        OpType::Max => Box::new(max_op::Max::new(inputs)),
        OpType::Min => Box::new(min_op::Min::new(inputs)),
        OpType::And => Box::new(and::And { inputs }),
        OpType::NonZero => Box::new(nonzero::NonZero::new(inputs)),
        OpType::Not => Box::new(not::Not::new(inputs)),
        OpType::PRelu => Box::new(prelu::PRelu::new(inputs)),
        OpType::Range => Box::new(range::Range::new(inputs)),
        OpType::Floor => Box::new(floor::Floor::new(inputs)),
        OpType::Sqrt => Box::new(sqrt::Sqrt::new(inputs)),
        OpType::ScatterElements => Box::new(scatter_elements::ScatterElements::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(0),
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::RoiAlign => {
            let mode = node
                .attrs
                .get_string("mode")
                .unwrap_or_else(|| "avg".to_string());
            let oh = node.attrs.get_int("output_height").unwrap_or(1) as usize;
            let ow = node.attrs.get_int("output_width").unwrap_or(1) as usize;
            let sr = node.attrs.get_int("sampling_ratio").unwrap_or(0) as usize;
            let ss = node.attrs.get_float("spatial_scale").unwrap_or(1.0);
            Box::new(roi_align::RoiAlign::new(inputs, mode, oh, ow, sr, ss))
        }
        OpType::ConstantOfShape => {
            let (fill_f32, fill_i64, dtype) = match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => match t.dtype() {
                    DType::Int64 => (
                        0.0,
                        t.ints().unwrap_or(&[]).first().copied().unwrap_or(0),
                        DType::Int64,
                    ),
                    _ => (
                        t.floats().unwrap_or(&[]).first().copied().unwrap_or(0.0),
                        0,
                        DType::Float,
                    ),
                },
                _ => (0.0, 0, DType::Float),
            };
            Box::new(constant_of_shape::ConstantOfShape::new(
                inputs, fill_f32, fill_i64, dtype,
            ))
        }
        OpType::Ceil => Box::new(ceil::Ceil::new(inputs)),
        OpType::Round => Box::new(round::Round::new(inputs)),
        OpType::Softmax => {
            // Softmax default axis changed in opset 13: was 1, now -1
            let default_axis = if opset_version >= 13 { -1 } else { 1 };
            Box::new(softmax::Softmax::new(
                inputs,
                node.attrs.get_int("axis").unwrap_or(default_axis),
                input_shapes[0],
            ))
        }
        OpType::Softplus => Box::new(softplus::Softplus::new(inputs)),
        OpType::Add => {
            let lb = node.attrs.get_int("broadcast").unwrap_or(0) != 0;
            let axis = node.attrs.get_int("axis").unwrap_or(0) as usize;
            Box::new(add::Add::new(inputs, lb, axis))
        }
        OpType::Sub => {
            let lb = node.attrs.get_int("broadcast").unwrap_or(0) != 0;
            let axis = node.attrs.get_int("axis").unwrap_or(0) as usize;
            Box::new(sub::Sub::new(inputs, lb, axis))
        }
        OpType::Mul => {
            let lb = node.attrs.get_int("broadcast").unwrap_or(0) != 0;
            let axis = node.attrs.get_int("axis").unwrap_or(0) as usize;
            Box::new(mul::Mul::new(inputs, lb, axis))
        }
        OpType::Div => {
            let lb = node.attrs.get_int("broadcast").unwrap_or(0) != 0;
            let axis = node.attrs.get_int("axis").unwrap_or(0) as usize;
            Box::new(div::Div::new(inputs, lb, axis))
        }
        OpType::Conv => {
            let ks = node.attrs.get_ints("kernel_shape").unwrap_or_default();
            let st = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
            let pa = node
                .attrs
                .get_ints("pads")
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = node
                .attrs
                .get_ints("dilations")
                .unwrap_or_else(|| vec![1, 1]);
            let gr = node.attrs.get_int("group").unwrap_or(1) as usize;
            let ap = node.attrs.get_string("auto_pad").unwrap_or_default();
            Box::new(conv::Conv::new(
                inputs,
                ks,
                st,
                pa,
                di,
                gr,
                ap,
                input_shapes[0],
                input_shapes[1],
            ))
        }
        OpType::MatMul => Box::new(matmul::MatMul::new(
            inputs,
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::Gemm => Box::new(gemm::Gemm::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(1.0),
            node.attrs.get_float("beta").unwrap_or(1.0),
            node.attrs.get_int("transA").unwrap_or(0) != 0,
            node.attrs.get_int("transB").unwrap_or(0) != 0,
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::MaxPool => {
            let ks = node.attrs.get_ints("kernel_shape").unwrap_or_default();
            let st = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
            let pa = node
                .attrs
                .get_ints("pads")
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let ap = node.attrs.get_string("auto_pad").unwrap_or_default();
            Box::new(maxpool::MaxPool::new(
                inputs,
                ks,
                st,
                pa,
                ap,
                input_shapes[0],
            )?)
        }
        OpType::AveragePool => {
            let ks = node.attrs.get_ints("kernel_shape").unwrap_or_default();
            let st = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
            let pa = node
                .attrs
                .get_ints("pads")
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let ap = node.attrs.get_string("auto_pad").unwrap_or_default();
            let cip = node.attrs.get_int("count_include_pad").unwrap_or(0);
            Box::new(average_pool::AveragePool::new(
                inputs,
                ks,
                st,
                pa,
                ap,
                cip,
                input_shapes[0],
            )?)
        }
        OpType::GlobalAveragePool => {
            Box::new(global_avg_pool::GlobalAvgPool::new(inputs, input_shapes[0]))
        }
        OpType::Flatten => Box::new(flatten::Flatten::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(1) as usize,
            input_shapes[0],
        )),
        OpType::Shape => Box::new(shape_op::Shape::new(inputs)),
        OpType::Gather => Box::new(gather::Gather::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(0),
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::Unsqueeze => Box::new(unsqueeze::Unsqueeze::new(
            inputs,
            node.attrs.get_ints("axes").unwrap_or_default(),
        )),
        OpType::Concat => Box::new(concat::Concat::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(0),
            &input_shapes,
        )),
        OpType::Identity => Box::new(identity::Identity::new(inputs)),
        OpType::Cast => Box::new(cast::Cast::new(
            inputs,
            node.attrs.get_int("to").unwrap_or(1),
        )),
        OpType::Transpose | OpType::LayoutTranspose => {
            let perm = node
                .attrs
                .get_ints("perm")
                .map(|p| p.iter().map(|&v| v as usize).collect());
            Box::new(transpose::Transpose::new(inputs, perm, input_shapes[0]))
        }
        OpType::Squeeze => Box::new(squeeze::Squeeze::new(
            inputs,
            node.attrs.get_ints("axes").unwrap_or_default(),
        )),
        OpType::Slice => {
            if let Some(attr_starts) = node.attrs.get_ints("starts") {
                let attr_ends = node.attrs.get_ints("ends").unwrap_or_default();
                let attr_axes = node.attrs.get_ints("axes");
                Box::new(slice::Slice::new_v1(
                    inputs,
                    attr_starts,
                    attr_ends,
                    attr_axes,
                ))
            } else {
                Box::new(slice::Slice::new(inputs))
            }
        }
        OpType::Tile => Box::new(tile::Tile::new(inputs)),
        OpType::Resize => {
            let ct = node
                .attrs
                .get_string("coordinate_transformation_mode")
                .unwrap_or_default();
            let nm = node.attrs.get_string("nearest_mode").unwrap_or_default();
            Box::new(resize::Resize::new(inputs, &ct, &nm))
        }
        OpType::Upsample => {
            let mode = node
                .attrs
                .get_string("mode")
                .unwrap_or_else(|| "nearest".to_string());
            let nm = if mode == "nearest" { "floor" } else { "" };
            let resize_inputs = vec![inputs[0].clone(), String::new(), inputs[1].clone()];
            Box::new(resize::Resize::new(resize_inputs, "asymmetric", nm))
        }
        OpType::Reshape => Box::new(reshape::Reshape::new(inputs, node.attrs.get_ints("shape"))),
        OpType::Constant => {
            let tensor = match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => t.clone(),
                _ => anyhow::bail!("Constant has no value"),
            };
            Box::new(constant::Constant::new(tensor))
        }
        OpType::ReduceMin => Box::new(reduce_min::ReduceMin::new(
            inputs,
            node.attrs.get_int("keepdims").unwrap_or(1) != 0,
            node.attrs.get_ints("axes"),
            input_shapes[0],
        )),
        OpType::NonMaxSuppression => Box::new(nms::Nms::new(inputs)),
        OpType::QuantizeLinear => Box::new(quantize_linear::QuantizeLinear::new(inputs)),
        OpType::DequantizeLinear => Box::new(dequantize_linear::DequantizeLinear::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(1),
            input_shapes[0],
        )),
        OpType::QLinearConv => {
            let has_bias = inputs.len() > 8 && !inputs[8].is_empty();
            let mut conv_inputs = vec!["__qconv_x__".to_string(), "__qconv_w__".to_string()];
            if has_bias {
                conv_inputs.push("__qconv_b__".to_string());
            }
            let ks = node.attrs.get_ints("kernel_shape").unwrap_or_default();
            let st = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
            let pa = node
                .attrs
                .get_ints("pads")
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = node
                .attrs
                .get_ints("dilations")
                .unwrap_or_else(|| vec![1, 1]);
            let gr = node.attrs.get_int("group").unwrap_or(1) as usize;
            let ap = node.attrs.get_string("auto_pad").unwrap_or_default();
            let inner = conv::Conv::new(
                conv_inputs,
                ks,
                st,
                pa,
                di,
                gr,
                ap,
                input_shapes[0],
                input_shapes[3],
            );
            Box::new(qlinear_conv::QLinearConv::new(inputs, inner))
        }
        OpType::QLinearAdd => Box::new(qlinear_add::QLinearAdd::new(inputs)),
        OpType::QLinearMatMul => {
            let inner = matmul::MatMul::new(
                vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()],
                input_shapes[0],
                input_shapes[3],
            );
            Box::new(qlinear_matmul::QLinearMatMul::new(inputs, inner))
        }
        OpType::QLinearGlobalAveragePool => {
            let inner = global_avg_pool::GlobalAvgPool::new(
                vec!["__qgap_x__".to_string()],
                input_shapes[0],
            );
            Box::new(qlinear_global_avg_pool::QLinearGlobalAvgPool::new(
                inputs, inner,
            ))
        }
        OpType::Abs => Box::new(abs::Abs::new(inputs)),
        OpType::ArgMax => Box::new(argmax::ArgMax::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(0),
            node.attrs.get_int("keepdims").unwrap_or(1) != 0,
            node.attrs.get_int("select_last_index").unwrap_or(0) != 0,
        )),
        OpType::CategoryMapper => {
            let cats_strings = match node.attrs.get("cats_strings") {
                Some(Attr::Strings(v)) => v.clone(),
                _ => vec![],
            };
            let cats_int64s = match node.attrs.get("cats_int64s") {
                Some(Attr::Ints(v)) => v.clone(),
                _ => vec![],
            };
            let default_int64 = node.attrs.get_int("default_int64").unwrap_or(-1);
            Box::new(category_mapper::CategoryMapper::new(
                inputs,
                cats_strings,
                cats_int64s,
                default_int64,
            ))
        }
        OpType::Compress => {
            let axis = node.attrs.get_int("axis");
            Box::new(compress::Compress::new(inputs, axis))
        }
        OpType::Dropout => Box::new(dropout::Dropout::new(inputs)),
        OpType::Hardmax => Box::new(hardmax::Hardmax::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(-1),
        )),
        OpType::Lstm => {
            let hs = node.attrs.get_int("hidden_size").unwrap_or(1) as usize;
            let dir_str = node.attrs.get_string("direction").unwrap_or_default();
            let direction = match dir_str.as_str() {
                "reverse" => lstm::LstmDirection::Reverse,
                "bidirectional" => lstm::LstmDirection::Bidirectional,
                _ => lstm::LstmDirection::Forward,
            };
            Box::new(lstm::Lstm::new(inputs, node.outputs.clone(), hs, direction))
        }
        OpType::ReduceMax => Box::new(reduce_max::ReduceMax::new(
            inputs,
            node.attrs.get_int("keepdims").unwrap_or(1) != 0,
            node.attrs.get_ints("axes"),
        )),
        OpType::ReduceMean => Box::new(reduce_mean::ReduceMean::new(
            inputs,
            node.attrs.get_int("keepdims").unwrap_or(1) != 0,
            node.attrs.get_ints("axes"),
            node.attrs.get_int("noop_with_empty_axes").unwrap_or(0) != 0,
        )),
        OpType::ReduceSum => Box::new(reduce_sum::ReduceSum::new(
            inputs,
            node.attrs.get_int("keepdims").unwrap_or(1) != 0,
            node.attrs.get_ints("axes"),
            node.attrs.get_int("noop_with_empty_axes").unwrap_or(0) != 0,
        )),
        OpType::Sum => Box::new(sum::Sum::new(inputs)),
        OpType::Where => Box::new(where_op::Where::new(inputs)),
        // Unary ops
        OpType::Sin => Box::new(unary_ops::Sin::new(inputs)),
        OpType::Cos => Box::new(unary_ops::Cos::new(inputs)),
        OpType::Tan => Box::new(unary_ops::Tan::new(inputs)),
        OpType::Asin => Box::new(unary_ops::Asin::new(inputs)),
        OpType::Acos => Box::new(unary_ops::Acos::new(inputs)),
        OpType::Atan => Box::new(unary_ops::Atan::new(inputs)),
        OpType::Sinh => Box::new(unary_ops::Sinh::new(inputs)),
        OpType::Cosh => Box::new(unary_ops::Cosh::new(inputs)),
        OpType::Asinh => Box::new(unary_ops::Asinh::new(inputs)),
        OpType::Acosh => Box::new(unary_ops::Acosh::new(inputs)),
        OpType::Atanh => Box::new(unary_ops::Atanh::new(inputs)),
        OpType::Erf => Box::new(unary_ops::Erf::new(inputs)),
        OpType::Sign => Box::new(unary_ops::Sign::new(inputs)),
        OpType::Neg => Box::new(unary_ops::Neg::new(inputs)),
        OpType::Reciprocal => Box::new(unary_ops::Reciprocal::new(inputs)),
        OpType::Softsign => Box::new(unary_ops::Softsign::new(inputs)),
        OpType::IsNaN => Box::new(unary_ops::IsNaN::new(inputs)),
        OpType::IsInf => Box::new(unary_ops::IsInf::new(inputs)),
        OpType::Elu => Box::new(unary_ops::Elu::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(1.0),
        )),
        OpType::Celu => Box::new(unary_ops::Celu::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(1.0),
        )),
        OpType::Selu => Box::new(unary_ops::Selu::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(1.673_263_2),
            node.attrs.get_float("gamma").unwrap_or(1.050_701),
        )),
        OpType::HardSigmoid => Box::new(unary_ops::HardSigmoid::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(0.2),
            node.attrs.get_float("beta").unwrap_or(0.5),
        )),
        OpType::ThresholdedRelu => Box::new(unary_ops::ThresholdedRelu::new(
            inputs,
            node.attrs.get_float("alpha").unwrap_or(1.0),
        )),
        OpType::Loop | OpType::Split | OpType::If | OpType::TopK | OpType::Scan => {
            unreachable!()
        }
    };

    Ok(PlanNode::Single { output, layer })
}

pub fn execute_node(node: &Node, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let op = node.op_type;

    let _span = tracing::trace_span!("op", op = %op, name = %node.name).entered();

    if op == OpType::Loop {
        let body = match node.attrs.get("body") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("Loop: no body graph"),
        };
        let mut loop_layer = loop_op::Loop::new(node.inputs.clone(), node.outputs.clone(), body);
        return loop_layer.execute(values);
    }

    if op == OpType::Split {
        let axis = node.attrs.get_int("axis").unwrap_or(0);
        let split_sizes = node.attrs.get_ints("split").unwrap_or_default();
        let mut split_layer =
            split::Split::new(node.inputs.clone(), node.outputs.clone(), axis, split_sizes);
        return split_layer.execute(values);
    }

    if op == OpType::If {
        let then_branch = match node.attrs.get("then_branch") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("If: no then_branch"),
        };
        let else_branch = match node.attrs.get("else_branch") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("If: no else_branch"),
        };
        let mut if_layer = if_op::If::new(
            node.inputs.clone(),
            node.outputs.clone(),
            then_branch,
            else_branch,
        );
        return if_layer.execute(values);
    }

    if op == OpType::TopK {
        let axis = node.attrs.get_int("axis").unwrap_or(-1);
        let largest = node.attrs.get_int("largest").unwrap_or(1) != 0;
        let mut topk_layer =
            topk::TopK::new(node.inputs.clone(), node.outputs.clone(), axis, largest);
        return topk_layer.execute(values);
    }

    if op == OpType::Scan {
        let body = match node.attrs.get("body") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => anyhow::bail!("Scan: no body graph"),
        };
        let num_scan_inputs = node.attrs.get_int("num_scan_inputs").unwrap_or(0) as usize;
        let scan_input_directions = node
            .attrs
            .get_ints("scan_input_directions")
            .unwrap_or_default();
        let scan_output_directions = node
            .attrs
            .get_ints("scan_output_directions")
            .unwrap_or_default();
        let mut scan_layer = scan::Scan::new(
            node.inputs.clone(),
            node.outputs.clone(),
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
        );
        return scan_layer.execute(values);
    }

    if node.outputs.is_empty() || node.outputs[0].is_empty() {
        return Ok(());
    }

    let expected = op.expected_input_dtypes();
    let mut to_cast: Vec<(usize, String)> = Vec::new();
    for (i, input_name) in node.inputs.iter().enumerate() {
        if input_name.is_empty() {
            continue;
        }
        if let Some(Some(expected_dt)) = expected.get(i) {
            if let Some(tensor) = values.get(input_name) {
                if tensor.dtype() != *expected_dt {
                    to_cast.push((i, input_name.clone()));
                }
            }
        }
    }

    let mut modified_inputs = node.inputs.clone();
    for (idx, (i, input_name)) in to_cast.into_iter().enumerate() {
        let cast_name = format!("__exec_cast_{idx}__");
        let src = values.get(&input_name).unwrap();
        let mut casted = Tensor::default();
        casted.copy_cast_f32(src).context("in plan auto-cast")?;
        values.insert(cast_name.clone(), casted);
        modified_inputs[i] = cast_name;
    }

    let exec_shape_map: HashMap<String, ShapeLayout> = values
        .iter()
        .map(|(k, v)| (k.clone(), ShapeLayout::new(v.dims.clone(), v.layout)))
        .collect();
    let mut plan_node = build_node(op, node, modified_inputs, &exec_shape_map)?;
    match &mut plan_node {
        PlanNode::Single { output, layer } => {
            let mut out = values.remove(output.as_str()).unwrap_or_default();
            let result = layer.execute(values, &mut out);
            values.insert(output.clone(), out);
            result
        }
        PlanNode::Loop(loop_layer) => loop_layer.execute(values),
        PlanNode::Split(split_layer) => split_layer.execute(values),
        PlanNode::If(if_layer) => if_layer.execute(values),
        PlanNode::TopK(topk_layer) => topk_layer.execute(values),
        PlanNode::Scan(scan_layer) => scan_layer.execute(values),
        #[cfg(feature = "xnnpack")]
        PlanNode::XnnpackSubgraph(_) => {
            anyhow::bail!("XnnpackSubgraph cannot be executed via execute_node")
        }
    }
}

#[cfg(feature = "xnnpack")]
fn compile_xnnpack_subgraphs(
    mut nodes: Vec<PlanNode>,
    node_meta: Vec<Option<(OpType, Vec<String>, Node)>>,
    shape_map: &mut HashMap<String, ShapeLayout>,
    type_map: &HashMap<String, DType>,
    initializers: &HashMap<String, Tensor>,
    _opset_version: i64,
    graph_output_names: &[String],
) -> Result<Vec<PlanNode>> {
    use super::xnnpack_subgraph::{CapturedOp, is_xnnpack_compatible};

    if std::env::var("XNNPACK_DISABLE").is_ok() {
        tracing::info!("XNNPACK disabled via XNNPACK_DISABLE env var");
        return Ok(nodes);
    }


    // Identify which plan nodes are XNNPACK-compatible
    let is_eligible = |idx: usize| -> bool {
        if let Some(Some((op, _inputs, node))) = node_meta.get(idx) {
            if !is_xnnpack_compatible(*op) {
                return false;
            }
            // Only float outputs
            for out in &node.outputs {
                if !out.is_empty() {
                    if let Some(&dt) = type_map.get(out) {
                        if dt != DType::Float {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    };

    // Find maximal runs of consecutive eligible ops
    let n = nodes.len();
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut i = 0;
    while i < n {
        if is_eligible(i) {
            let start = i;
            while i < n && is_eligible(i) {
                i += 1;
            }
            if i - start >= 2 {
                runs.push(start..i);
            }
        } else {
            i += 1;
        }
    }

    // Process runs in reverse order (to preserve indices)
    for run in runs.into_iter().rev() {
        // Identify which outputs are consumed outside this run
        let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut consumed_after: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for idx in run.clone() {
            if let Some(Some((_, _, node))) = node_meta.get(idx) {
                for out in &node.outputs {
                    produced.insert(out.clone());
                }
            }
        }

        // Check what's consumed after the run
        for idx in run.end..n {
            if let Some(Some((_, inputs, _))) = node_meta.get(idx) {
                for inp in inputs {
                    if !inp.is_empty() && produced.contains(inp) {
                        consumed_after.insert(inp.clone());
                    }
                }
            }
        }

        // Include any produced value that is consumed after the subgraph
        // or is a graph output
        let graph_output_set: std::collections::HashSet<&str> =
            graph_output_names.iter().map(|s| s.as_str()).collect();

        let required_outputs: Vec<String> = produced
            .iter()
            .filter(|name| {
                consumed_after.contains(name.as_str())
                    || graph_output_set.contains(name.as_str())
            })
            .cloned()
            .collect();

        if required_outputs.is_empty() {
            continue;
        }

        // Capture ops
        let captured: Vec<CapturedOp> = node_meta[run.clone()]
            .iter()
            .filter_map(|meta| {
                let (op, inputs, node) = meta.as_ref()?;
                Some(CapturedOp {
                    op: *op,
                    inputs: inputs.clone(),
                    outputs: node.outputs.clone(),
                    node: node.clone(),
                })
            })
            .collect();

        // Collect initializers referenced by these ops
        let mut sub_initializers: HashMap<String, Tensor> = HashMap::new();
        for cap in &captured {
            for inp in &cap.inputs {
                if !inp.is_empty() && !sub_initializers.contains_key(inp) {
                    if let Some(tensor) = initializers.get(inp) {
                        sub_initializers.insert(inp.clone(), tensor.clone());
                    }
                }
            }
        }

        // Build shape hints for XNNPACK from the layout-aware shape_map
        let shape_map_vec: HashMap<String, Vec<usize>> = shape_map
            .iter()
            .map(|(k, v)| (k.clone(), v.dims.to_vec()))
            .collect();

        // Remove the CPU plan nodes that are now covered by the XNNPACK subgraph
        nodes.drain(run.clone());

        let subgraph = super::xnnpack_subgraph::XnnpackSubgraph::new(
            captured,
            required_outputs,
            shape_map_vec,
            sub_initializers,
        );

        nodes.insert(run.start, PlanNode::XnnpackSubgraph(Box::new(subgraph)));
    }

    Ok(nodes)
}

/// Infer the output layout for an op based on its type and input layouts.
///
/// - `LayoutTranspose`: explicitly changes layout (NCHW↔NHWC)
/// - Regular `Transpose` on 4D data: degrades to `Unknown`
/// - Layout-preserving ops (unary, binary, pooling, conv): propagate input layout
/// - Rank-changing ops (Reshape, Flatten, Squeeze, Unsqueeze, Gemm, MatMul): `Unknown`
fn infer_output_layout(
    op: OpType,
    node: &crate::onnx_ir::Node,
    shape_map: &HashMap<String, ShapeLayout>,
) -> Layout {
    if op == OpType::LayoutTranspose {
        let perm = node.attrs.get_ints("perm").unwrap_or_default();
        if perm == [0, 2, 3, 1] {
            return Layout::NHWC;
        } else if perm == [0, 3, 1, 2] {
            return Layout::NCHW;
        }
        return Layout::Unknown;
    }

    // Regular Transpose degrades layout to Unknown
    if op == OpType::Transpose {
        return Layout::Unknown;
    }

    // Rank-changing ops degrade to Unknown
    if matches!(
        op,
        OpType::Reshape
            | OpType::Flatten
            | OpType::Squeeze
            | OpType::Unsqueeze
            | OpType::Gemm
            | OpType::MatMul
    ) {
        return Layout::Unknown;
    }

    // For all other ops, propagate layout from first data input
    node.inputs
        .first()
        .filter(|s| !s.is_empty())
        .and_then(|name| shape_map.get(name))
        .map(|sl| sl.layout)
        .unwrap_or(Layout::NCHW)
}
