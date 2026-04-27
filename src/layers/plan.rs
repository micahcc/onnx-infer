use std::collections::HashMap;
use std::sync::Arc;

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
use crate::layers::conv_transpose;
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
#[cfg(feature = "xnnpack")]
use crate::layers::xnnpack_subgraph;
use crate::onnx_ir::Graph;
use crate::onnx_ir::Node;
use crate::onnx_ir::NodeOp;

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
    XnnpackSubgraph(Box<xnnpack_subgraph::XnnpackSubgraph>),
}

pub struct Plan {
    pub nodes: Vec<PlanNode>,
    pub output_names: Vec<String>,
    pub shape_map: HashMap<String, ShapeLayout>,
    pub type_map: HashMap<String, DType>,
    /// Constants folded during plan building. These may depend on input
    /// values/shapes and are specific to this plan, so they live here
    /// rather than in the shared initializers map. Wrapped in Arc so
    /// execution can cheaply hold a reference while mutating the plan.
    pub folded: Arc<HashMap<String, Tensor>>,
}

impl Plan {
    pub fn build(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::build_full(graph, input_sizes, &HashMap::new(), initializers)
    }

    pub fn build_with_types(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::build_full(graph, input_sizes, type_hints, initializers)
    }

    #[cfg(feature = "xnnpack")]
    pub fn build_with_xnnpack(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::build_full_inner(
            graph,
            input_sizes,
            &HashMap::new(),
            true,
            initializers,
            None,
        )
    }

    #[cfg(feature = "xnnpack")]
    pub fn build_with_xnnpack_quant(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        initializers: &HashMap<String, Tensor>,
        quant_map: crate::quant_patterns::QuantMap,
    ) -> Result<Self> {
        Self::build_full_inner(
            graph,
            input_sizes,
            &HashMap::new(),
            true,
            initializers,
            Some(quant_map),
        )
    }

    pub fn build_full(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
        initializers: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::build_full_inner(graph, input_sizes, type_hints, false, initializers, None)
    }

    #[allow(unused_variables)]
    fn build_full_inner(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
        enable_xnnpack: bool,
        initializers: &HashMap<String, Tensor>,
        quant_map: Option<crate::quant_patterns::QuantMap>,
    ) -> Result<Self> {
        let output_names: Vec<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();

        let mut type_map: HashMap<String, DType> = HashMap::new();
        for (name, tensor) in initializers.iter() {
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
        // Layout-only map for ops whose shapes aren't known but layout is.
        let mut layout_only: HashMap<String, Layout> = HashMap::new();
        for (name, tensor) in initializers.iter() {
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
                // Override with user-provided dims (supports dynamic batch)
                existing.dims = user_dims.clone();
            } else {
                shape_map.insert(name.clone(), ShapeLayout::nchw(user_dims.clone()));
            }
        }

        let mut known_values: HashMap<String, Tensor> = HashMap::new();
        let mut folded: HashMap<String, Tensor> = HashMap::new();

        let mut nodes = Vec::new();
        #[cfg(feature = "xnnpack")]
        let mut node_meta: Vec<Option<(OpType, Vec<String>, Node)>> = Vec::new();
        let mut cast_counter = 0usize;
        for node in &graph.nodes {
            let op = node.op_type();

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
                initializers,
                &shape_map,
            ) {
                if let Some(out_name) = out_name {
                    tensor.layout = infer_output_layout(op, node, &shape_map, &layout_only);
                    shape_map.insert(
                        out_name.clone(),
                        ShapeLayout::new(tensor.dims.clone(), tensor.layout),
                    );
                    folded.insert(out_name.clone(), tensor.clone());
                    known_values.insert(out_name.clone(), tensor);
                }
                continue;
            }
            if let Some(shape) =
                op.infer_output_shape(node, &node.inputs, &shape_map, &known_values, initializers)
            {
                if let Some(out_name) = out_name {
                    let out_layout = infer_output_layout(op, node, &shape_map, &layout_only);
                    shape_map.insert(out_name.clone(), ShapeLayout::new(shape, out_layout));
                }
            } else {
                // Even without shape, record layout so downstream ops know their input layout.
                if let Some(out_name) = out_name {
                    let out_layout = infer_output_layout(op, node, &shape_map, &layout_only);
                    if out_layout != Layout::Unknown {
                        layout_only.insert(out_name.clone(), out_layout);
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
                    let axis_attr = match &node.op {
                        NodeOp::Split { axis } => *axis,
                        _ => 0,
                    };
                    let rank = in_sl.dims.len() as i64;
                    let axis = if axis_attr < 0 {
                        (rank + axis_attr) as usize
                    } else {
                        axis_attr as usize
                    };
                    let split_sizes: Option<Vec<i64>> = node
                        .inputs
                        .get(1)
                        .filter(|s| !s.is_empty())
                        .and_then(|name| known_values.get(name).or_else(|| initializers.get(name)))
                        .and_then(|t| t.ints().ok().map(|s| s.to_vec()));
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
                        shape_map
                            .insert(out_name.clone(), ShapeLayout::new(out_shape, in_sl.layout));
                    }
                }
            }

            // Try to fold Loop ops when all inputs are known and none are tainted
            if op == OpType::Loop {
                let all_inputs_known = node.inputs.iter().all(|n| {
                    n.is_empty() || known_values.contains_key(n) || initializers.contains_key(n)
                });

                if all_inputs_known {
                    let plan_node = build_node_with_initializers(
                        op,
                        node,
                        modified_inputs.clone(),
                        &shape_map,
                        &layout_only,
                        initializers,
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
                        if let Ok(()) =
                            loop_layer.execute(&mut temp_values, &crate::Constants::empty())
                        {
                            let mut loop_folded = true;
                            for out_name in &node.outputs {
                                if !out_name.is_empty() {
                                    if let Some(t) = temp_values.remove(out_name) {
                                        shape_map.insert(
                                            out_name.clone(),
                                            ShapeLayout::new(t.dims.clone(), t.layout),
                                        );
                                        folded.insert(out_name.clone(), t.clone());
                                        known_values.insert(out_name.clone(), t);
                                    } else {
                                        loop_folded = false;
                                    }
                                }
                            }
                            if loop_folded {
                                continue;
                            }
                        }
                    }
                }
            }

            #[cfg(feature = "xnnpack")]
            node_meta.push(Some((op, modified_inputs.clone(), node.clone())));
            nodes.push(build_node_with_initializers(
                op,
                node,
                modified_inputs,
                &shape_map,
                &layout_only,
                initializers,
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
                &output_names,
                quant_map,
            )?;
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
                    tracing::info!(
                        "plan: {total} ops, {xnnpack_ops} XNNPACK ({pct}%), {cpu_ops} CPU"
                    );
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
            output_names,
            shape_map,
            type_map,
            folded: Arc::new(folded),
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
        // Only fold Shape if the input is a known constant (initializer or previously folded).
        // Shape of runtime tensors must be computed at runtime because shape inference
        // can be inaccurate for dynamic dimensions (e.g. spatial dims in detection models).
        let name = input_names.first().filter(|s| !s.is_empty())?;
        let tensor = known_values.get(name).or_else(|| initializers.get(name))?;
        let dims: Vec<i64> = tensor.dims.iter().map(|&d| d as i64).collect();
        return Some(Tensor::new_i64(dims![dims.len()], dims));
    }

    if op == OpType::Constant {
        return match &node.op {
            NodeOp::Constant { value } => Some(value.clone()),
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
        let vals = crate::Values {
            intermediates: &temp_values,
            constants: crate::Constants::empty(),
        };
        layer.execute(&vals, &mut output).ok()?;
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
    build_node_with_initializers(
        op,
        node,
        inputs,
        shape_map,
        &HashMap::new(),
        &HashMap::new(),
    )
}

pub fn build_node_with_initializers(
    op: OpType,
    node: &Node,
    inputs: Vec<String>,
    shape_map: &HashMap<String, ShapeLayout>,
    layout_only: &HashMap<String, Layout>,
    initializers: &HashMap<String, Tensor>,
) -> Result<PlanNode> {
    if op == OpType::Loop {
        let body = match &node.op {
            NodeOp::Loop { body } => (**body).clone(),
            _ => anyhow::bail!("Loop: no body graph"),
        };
        return Ok(PlanNode::Loop(Box::new(loop_op::Loop::new(
            inputs,
            node.outputs.clone(),
            body,
        ))));
    }

    if op == OpType::Split {
        let NodeOp::Split { axis } = &node.op else {
            unreachable!()
        };
        let split_sizes: Vec<i64> = inputs
            .get(1)
            .filter(|s| !s.is_empty())
            .and_then(|name| initializers.get(name))
            .and_then(|t| t.ints().ok().map(|s| s.to_vec()))
            .unwrap_or_default();
        return Ok(PlanNode::Split(Box::new(split::Split::new(
            inputs,
            node.outputs.clone(),
            *axis,
            split_sizes,
        ))));
    }

    if op == OpType::If {
        let NodeOp::If {
            then_branch,
            else_branch,
        } = &node.op
        else {
            anyhow::bail!("If: expected NodeOp::If");
        };
        return Ok(PlanNode::If(Box::new(if_op::If::new(
            inputs,
            node.outputs.clone(),
            (**then_branch).clone(),
            (**else_branch).clone(),
        ))));
    }

    if op == OpType::TopK {
        let NodeOp::TopK { axis, largest } = &node.op else {
            unreachable!()
        };
        return Ok(PlanNode::TopK(Box::new(topk::TopK::new(
            inputs,
            node.outputs.clone(),
            *axis,
            *largest,
        ))));
    }

    if op == OpType::Scan {
        let NodeOp::Scan {
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
        } = &node.op
        else {
            anyhow::bail!("Scan: expected NodeOp::Scan");
        };
        return Ok(PlanNode::Scan(Box::new(scan::Scan::new(
            inputs,
            node.outputs.clone(),
            (**body).clone(),
            *num_scan_inputs as usize,
            scan_input_directions.clone(),
            scan_output_directions.clone(),
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

    let input_layout = |idx: usize| -> Layout {
        inputs
            .get(idx)
            .filter(|s| !s.is_empty())
            .and_then(|name| {
                shape_map
                    .get(name)
                    .map(|sl| sl.layout)
                    .or_else(|| layout_only.get(name).copied())
            })
            .unwrap_or(Layout::NCHW)
    };

    let layer: Box<dyn Layer> =
        match &node.op {
            NodeOp::Relu => Box::new(relu::Relu::new(inputs)),
            NodeOp::LeakyRelu { alpha } => Box::new(leaky_relu::LeakyRelu::new(inputs, *alpha)),
            NodeOp::Clip => Box::new(clip::Clip::new(inputs, f32::NEG_INFINITY, f32::INFINITY)),
            NodeOp::BatchNormalization { epsilon } => Box::new(batch_norm::BatchNorm::new(
                inputs,
                *epsilon,
                input_shapes[0],
                false,
            )),
            NodeOp::BatchNormalization2d { epsilon } => Box::new(batch_norm::BatchNorm::new(
                inputs,
                *epsilon,
                input_shapes[0],
                true,
            )),
            NodeOp::Sigmoid => Box::new(sigmoid::Sigmoid::new(inputs)),
            NodeOp::Exp => Box::new(exp::Exp::new(inputs)),
            NodeOp::Log => Box::new(log::Log::new(inputs)),
            NodeOp::Lrn {
                size,
                alpha,
                beta,
                bias,
            } => Box::new(lrn::Lrn::new(inputs, *size as usize, *alpha, *beta, *bias)),
            NodeOp::Tanh => Box::new(tanh::Tanh::new(inputs)),
            NodeOp::Expand => Box::new(expand::Expand::new(inputs)),
            NodeOp::Less => Box::new(less::Less::new(inputs)),
            NodeOp::Equal => Box::new(equal::Equal::new(inputs)),
            NodeOp::Greater => Box::new(greater::Greater::new(inputs)),
            NodeOp::Max => Box::new(max_op::Max::new(inputs)),
            NodeOp::Min => Box::new(min_op::Min::new(inputs)),
            NodeOp::And => Box::new(and::And { inputs }),
            NodeOp::NonZero => Box::new(nonzero::NonZero::new(inputs)),
            NodeOp::Not => Box::new(not::Not::new(inputs)),
            NodeOp::PRelu => Box::new(prelu::PRelu::new(inputs)),
            NodeOp::Range => Box::new(range::Range::new(inputs)),
            NodeOp::Floor => Box::new(floor::Floor::new(inputs)),
            NodeOp::Sqrt => Box::new(sqrt::Sqrt::new(inputs)),
            NodeOp::ScatterElements { axis } => Box::new(scatter_elements::ScatterElements::new(
                inputs,
                *axis,
                input_shapes[0],
                input_shapes[1],
            )),
            NodeOp::RoiAlign {
                mode,
                output_height,
                output_width,
                sampling_ratio,
                spatial_scale,
            } => Box::new(roi_align::RoiAlign::new(
                inputs,
                mode.clone(),
                *output_height as usize,
                *output_width as usize,
                *sampling_ratio as usize,
                *spatial_scale,
            )),
            NodeOp::ConstantOfShape { value } => {
                let (fill_f32, fill_i64, dtype) = match value {
                    Some(t) => match t.dtype() {
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
                    None => (0.0, 0, DType::Float),
                };
                Box::new(constant_of_shape::ConstantOfShape::new(
                    inputs, fill_f32, fill_i64, dtype,
                ))
            }
            NodeOp::Ceil => Box::new(ceil::Ceil::new(inputs)),
            NodeOp::Round => Box::new(round::Round::new(inputs)),
            NodeOp::Softmax { axis, coerce_2d } => Box::new(softmax::Softmax::new(
                inputs,
                *axis,
                *coerce_2d,
                input_shapes[0],
            )),
            NodeOp::Softplus => Box::new(softplus::Softplus::new(inputs)),
            NodeOp::Add {
                legacy_broadcast,
                axis,
            } => Box::new(add::Add::new(inputs, *legacy_broadcast, *axis as usize)),
            NodeOp::Sub {
                legacy_broadcast,
                axis,
            } => Box::new(sub::Sub::new(inputs, *legacy_broadcast, *axis as usize)),
            NodeOp::Mul {
                legacy_broadcast,
                axis,
            } => Box::new(mul::Mul::new(inputs, *legacy_broadcast, *axis as usize)),
            NodeOp::Div {
                legacy_broadcast,
                axis,
            } => Box::new(div::Div::new(inputs, *legacy_broadcast, *axis as usize)),
            NodeOp::Conv {
                kernel_shape,
                strides,
                pads,
                dilations,
                group,
                auto_pad,
            } => {
                let nhwc = input_layout(0) == Layout::NHWC;
                Box::new(conv::Conv::new(
                    inputs,
                    kernel_shape.clone(),
                    strides.clone(),
                    pads.clone(),
                    dilations.clone(),
                    *group as usize,
                    auto_pad.clone(),
                    input_shapes[0],
                    input_shapes[1],
                    nhwc,
                ))
            }
            NodeOp::ConvTranspose {
                strides,
                pads,
                dilations,
                group,
            } => Box::new(conv_transpose::ConvTranspose::new(
                inputs, strides, pads, dilations, *group,
            )),
            NodeOp::MatMul => Box::new(matmul::MatMul::new(
                inputs,
                input_shapes[0],
                input_shapes[1],
            )),
            NodeOp::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
            } => Box::new(gemm::Gemm::new(
                inputs,
                *alpha,
                *beta,
                *trans_a,
                *trans_b,
                input_shapes[0],
                input_shapes[1],
            )),
            NodeOp::MaxPool {
                kernel_shape,
                strides,
                pads,
                auto_pad,
            } => {
                let nhwc = input_layout(0) == Layout::NHWC;
                Box::new(maxpool::MaxPool::new(
                    inputs,
                    kernel_shape.clone(),
                    strides.clone(),
                    pads.clone(),
                    auto_pad.clone(),
                    input_shapes[0],
                    nhwc,
                )?)
            }
            NodeOp::AveragePool {
                kernel_shape,
                strides,
                pads,
                auto_pad,
                count_include_pad,
            } => {
                let nhwc = input_layout(0) == Layout::NHWC;
                Box::new(average_pool::AveragePool::new(
                    inputs,
                    kernel_shape.clone(),
                    strides.clone(),
                    pads.clone(),
                    auto_pad.clone(),
                    *count_include_pad,
                    input_shapes[0],
                    nhwc,
                )?)
            }
            NodeOp::GlobalAveragePool => {
                let nhwc = input_layout(0) == Layout::NHWC;
                Box::new(global_avg_pool::GlobalAvgPool::new(
                    inputs,
                    input_shapes[0],
                    nhwc,
                ))
            }
            NodeOp::Flatten { axis } => Box::new(flatten::Flatten::new(
                inputs,
                *axis as usize,
                input_shapes[0],
            )),
            NodeOp::Shape => Box::new(shape_op::Shape::new(inputs)),
            NodeOp::Gather { axis } => Box::new(gather::Gather::new(
                inputs,
                *axis,
                input_shapes[0],
                input_shapes[1],
            )),
            NodeOp::Unsqueeze => Box::new(unsqueeze::Unsqueeze::new(inputs, vec![])),
            NodeOp::Concat { axis } => Box::new(concat::Concat::new(inputs, *axis, &input_shapes)),
            NodeOp::Identity => Box::new(identity::Identity::new(inputs)),
            NodeOp::Cast { to } => Box::new(cast::Cast::new(inputs, *to)),
            NodeOp::Transpose { .. } | NodeOp::LayoutTranspose { .. } => {
                let perm = node
                    .op
                    .perm()
                    .map(|p| p.iter().map(|&v| v as usize).collect());
                Box::new(transpose::Transpose::new(inputs, perm, input_shapes[0]))
            }
            NodeOp::Squeeze => Box::new(squeeze::Squeeze::new(inputs, vec![])),
            NodeOp::Slice => Box::new(slice::Slice::new(inputs)),
            NodeOp::Tile => Box::new(tile::Tile::new(inputs)),
            NodeOp::Resize {
                mode,
                coordinate_transformation_mode,
                nearest_mode,
            } => {
                let nhwc = input_layout(0) == Layout::NHWC;
                Box::new(resize::Resize::new(
                    inputs,
                    mode,
                    coordinate_transformation_mode,
                    nearest_mode,
                    nhwc,
                ))
            }
            NodeOp::Upsample { mode } => {
                let nm = if mode == "nearest" { "floor" } else { "" };
                let resize_inputs = vec![inputs[0].clone(), String::new(), inputs[1].clone()];
                let nhwc = input_layout(0) == Layout::NHWC;
                Box::new(resize::Resize::new(
                    resize_inputs,
                    mode,
                    "asymmetric",
                    nm,
                    nhwc,
                ))
            }
            NodeOp::Reshape => Box::new(reshape::Reshape::new(inputs, None)),
            NodeOp::Constant { value } => Box::new(constant::Constant::new(value.clone())),
            NodeOp::ReduceMin { keepdims } => Box::new(reduce_min::ReduceMin::new(
                inputs,
                *keepdims,
                None,
                input_shapes[0],
            )),
            NodeOp::NonMaxSuppression => Box::new(nms::Nms::new(inputs)),
            NodeOp::QuantizeLinear => Box::new(quantize_linear::QuantizeLinear::new(inputs)),
            NodeOp::DequantizeLinear { axis } => Box::new(
                dequantize_linear::DequantizeLinear::new(inputs, *axis, input_shapes[0]),
            ),
            NodeOp::QLinearConv {
                kernel_shape,
                strides,
                pads,
                dilations,
                group,
                auto_pad,
            } => {
                let has_bias = inputs.len() > 8 && !inputs[8].is_empty();
                let mut conv_inputs = vec!["__qconv_x__".to_string(), "__qconv_w__".to_string()];
                if has_bias {
                    conv_inputs.push("__qconv_b__".to_string());
                }
                let inner = conv::Conv::new(
                    conv_inputs,
                    kernel_shape.clone(),
                    strides.clone(),
                    pads.clone(),
                    dilations.clone(),
                    *group as usize,
                    auto_pad.clone(),
                    input_shapes[0],
                    input_shapes[3],
                    input_layout(0) == Layout::NHWC,
                );
                Box::new(qlinear_conv::QLinearConv::new(inputs, inner))
            }
            NodeOp::QLinearAdd => Box::new(qlinear_add::QLinearAdd::new(inputs)),
            NodeOp::QLinearMatMul => {
                let inner = matmul::MatMul::new(
                    vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()],
                    input_shapes[0],
                    input_shapes[3],
                );
                Box::new(qlinear_matmul::QLinearMatMul::new(inputs, inner))
            }
            NodeOp::QLinearGlobalAveragePool => {
                let inner = global_avg_pool::GlobalAvgPool::new(
                    vec!["__qgap_x__".to_string()],
                    input_shapes[0],
                    input_layout(0) == Layout::NHWC,
                );
                Box::new(qlinear_global_avg_pool::QLinearGlobalAvgPool::new(
                    inputs, inner,
                ))
            }
            NodeOp::Abs => Box::new(abs::Abs::new(inputs)),
            NodeOp::ArgMax {
                axis,
                keepdims,
                select_last_index,
            } => Box::new(argmax::ArgMax::new(
                inputs,
                *axis,
                *keepdims,
                *select_last_index,
            )),
            NodeOp::CategoryMapper {
                cats_strings,
                cats_int64s,
                default_int64,
            } => Box::new(category_mapper::CategoryMapper::new(
                inputs,
                cats_strings.clone(),
                cats_int64s.clone(),
                *default_int64,
            )),
            NodeOp::Compress { axis } => Box::new(compress::Compress::new(inputs, *axis)),
            NodeOp::Dropout => Box::new(dropout::Dropout::new(inputs)),
            NodeOp::Hardmax { axis } => Box::new(hardmax::Hardmax::new(inputs, *axis)),
            NodeOp::Lstm {
                hidden_size,
                direction,
            } => {
                let dir = match direction.as_str() {
                    "reverse" => lstm::LstmDirection::Reverse,
                    "bidirectional" => lstm::LstmDirection::Bidirectional,
                    _ => lstm::LstmDirection::Forward,
                };
                Box::new(lstm::Lstm::new(
                    inputs,
                    node.outputs.clone(),
                    *hidden_size as usize,
                    dir,
                ))
            }
            NodeOp::ReduceMax { keepdims } => {
                Box::new(reduce_max::ReduceMax::new(inputs, *keepdims, None))
            }
            NodeOp::ReduceMean {
                keepdims,
                noop_with_empty_axes,
            } => Box::new(reduce_mean::ReduceMean::new(
                inputs,
                *keepdims,
                None,
                *noop_with_empty_axes,
            )),
            NodeOp::ReduceSum {
                keepdims,
                noop_with_empty_axes,
            } => Box::new(reduce_sum::ReduceSum::new(
                inputs,
                *keepdims,
                None,
                *noop_with_empty_axes,
            )),
            NodeOp::Sum => Box::new(sum::Sum::new(inputs)),
            NodeOp::Where => Box::new(where_op::Where::new(inputs)),
            // Unary ops
            NodeOp::Sin => Box::new(unary_ops::Sin::new(inputs)),
            NodeOp::Cos => Box::new(unary_ops::Cos::new(inputs)),
            NodeOp::Tan => Box::new(unary_ops::Tan::new(inputs)),
            NodeOp::Asin => Box::new(unary_ops::Asin::new(inputs)),
            NodeOp::Acos => Box::new(unary_ops::Acos::new(inputs)),
            NodeOp::Atan => Box::new(unary_ops::Atan::new(inputs)),
            NodeOp::Sinh => Box::new(unary_ops::Sinh::new(inputs)),
            NodeOp::Cosh => Box::new(unary_ops::Cosh::new(inputs)),
            NodeOp::Asinh => Box::new(unary_ops::Asinh::new(inputs)),
            NodeOp::Acosh => Box::new(unary_ops::Acosh::new(inputs)),
            NodeOp::Atanh => Box::new(unary_ops::Atanh::new(inputs)),
            NodeOp::Erf => Box::new(unary_ops::Erf::new(inputs)),
            NodeOp::Sign => Box::new(unary_ops::Sign::new(inputs)),
            NodeOp::Neg => Box::new(unary_ops::Neg::new(inputs)),
            NodeOp::Reciprocal => Box::new(unary_ops::Reciprocal::new(inputs)),
            NodeOp::Softsign => Box::new(unary_ops::Softsign::new(inputs)),
            NodeOp::IsNaN => Box::new(unary_ops::IsNaN::new(inputs)),
            NodeOp::IsInf => Box::new(unary_ops::IsInf::new(inputs)),
            NodeOp::Elu { alpha } => Box::new(unary_ops::Elu::new(inputs, *alpha)),
            NodeOp::Celu { alpha } => Box::new(unary_ops::Celu::new(inputs, *alpha)),
            NodeOp::Selu { alpha, gamma } => Box::new(unary_ops::Selu::new(inputs, *alpha, *gamma)),
            NodeOp::HardSigmoid { alpha, beta } => {
                Box::new(unary_ops::HardSigmoid::new(inputs, *alpha, *beta))
            }
            NodeOp::ThresholdedRelu { alpha } => {
                Box::new(unary_ops::ThresholdedRelu::new(inputs, *alpha))
            }
            NodeOp::Loop { .. }
            | NodeOp::Split { .. }
            | NodeOp::If { .. }
            | NodeOp::TopK { .. }
            | NodeOp::Scan { .. } => {
                unreachable!("multi-output ops handled above")
            }
        };

    Ok(PlanNode::Single { output, layer })
}

pub fn execute_node(node: &Node, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let op = node.op_type();

    let _span = tracing::trace_span!("op", op = %op, name = %node.name).entered();

    if op == OpType::Loop {
        let body = match &node.op {
            NodeOp::Loop { body } => (**body).clone(),
            _ => anyhow::bail!("Loop: no body graph"),
        };
        let mut loop_layer = loop_op::Loop::new(node.inputs.clone(), node.outputs.clone(), body);
        return loop_layer.execute(values, &crate::Constants::empty());
    }

    if op == OpType::Split {
        let NodeOp::Split { axis } = &node.op else {
            unreachable!()
        };
        let split_sizes: Vec<i64> = node
            .inputs
            .get(1)
            .filter(|s| !s.is_empty())
            .and_then(|name| values.get(name))
            .and_then(|t| t.ints().ok().map(|s| s.to_vec()))
            .unwrap_or_default();
        let mut split_layer = split::Split::new(
            node.inputs.clone(),
            node.outputs.clone(),
            *axis,
            split_sizes,
        );
        return split_layer.execute(values, &crate::Constants::empty());
    }

    if op == OpType::If {
        let NodeOp::If {
            then_branch,
            else_branch,
        } = &node.op
        else {
            anyhow::bail!("If: expected NodeOp::If");
        };
        let mut if_layer = if_op::If::new(
            node.inputs.clone(),
            node.outputs.clone(),
            (**then_branch).clone(),
            (**else_branch).clone(),
        );
        return if_layer.execute(values, &crate::Constants::empty());
    }

    if op == OpType::TopK {
        let NodeOp::TopK { axis, largest } = &node.op else {
            unreachable!()
        };
        let mut topk_layer =
            topk::TopK::new(node.inputs.clone(), node.outputs.clone(), *axis, *largest);
        return topk_layer.execute(values, &crate::Constants::empty());
    }

    if op == OpType::Scan {
        let NodeOp::Scan {
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
        } = &node.op
        else {
            anyhow::bail!("Scan: expected NodeOp::Scan");
        };
        let mut scan_layer = scan::Scan::new(
            node.inputs.clone(),
            node.outputs.clone(),
            (**body).clone(),
            *num_scan_inputs as usize,
            scan_input_directions.clone(),
            scan_output_directions.clone(),
        );
        return scan_layer.execute(values, &crate::Constants::empty());
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
            // Capture input layout before borrowing values for execute.
            let input_layout = node
                .inputs
                .first()
                .and_then(|name| values.get(name))
                .map(|t| t.layout)
                .unwrap_or(Layout::NCHW);
            let vals = crate::Values {
                intermediates: values,
                constants: crate::Constants::empty(),
            };
            let result = layer.execute(&vals, &mut out);
            // Propagate layout through the op chain so downstream ops see the
            // correct layout in the shape map built from values.
            out.layout = if op == OpType::LayoutTranspose {
                let perm = node.op.perm().unwrap_or(&[]);
                match (input_layout, perm) {
                    (Layout::NCHW, [0, 2, 3, 1]) => Layout::NHWC,
                    (Layout::NHWC, [0, 3, 1, 2]) => Layout::NCHW,
                    _ => Layout::Unknown,
                }
            } else {
                infer_output_layout(op, node, &exec_shape_map, &HashMap::new())
            };
            values.insert(output.clone(), out);
            result
        }
        PlanNode::Loop(loop_layer) => loop_layer.execute(values, &crate::Constants::empty()),
        PlanNode::Split(split_layer) => split_layer.execute(values, &crate::Constants::empty()),
        PlanNode::If(if_layer) => if_layer.execute(values, &crate::Constants::empty()),
        PlanNode::TopK(topk_layer) => topk_layer.execute(values, &crate::Constants::empty()),
        PlanNode::Scan(scan_layer) => scan_layer.execute(values, &crate::Constants::empty()),
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
    graph_output_names: &[String],
    quant_map: Option<crate::quant_patterns::QuantMap>,
) -> Result<Vec<PlanNode>> {
    use xnnpack_subgraph::CapturedOp;
    use xnnpack_subgraph::is_xnnpack_compatible;

    if std::env::var("XNNPACK_DISABLE").is_ok() {
        tracing::info!("XNNPACK disabled via XNNPACK_DISABLE env var");
        return Ok(nodes);
    }

    let has_quant = quant_map.is_some();

    // Identify which plan nodes are XNNPACK-compatible
    let is_eligible = |idx: usize| -> bool {
        if let Some(Some((op, _inputs, node))) = node_meta.get(idx) {
            if !is_xnnpack_compatible(*op) {
                return false;
            }
            // XNNPACK only supports bilinear resize, not nearest
            if *op == OpType::Resize {
                let mode = match &node.op {
                    NodeOp::Resize { mode, .. } => mode.as_str(),
                    _ => "",
                };
                if mode != "linear" {
                    return false;
                }
            }
            // Allow quantized ops if we have a quant_map
            let is_quant_op = matches!(
                op,
                OpType::DequantizeLinear
                    | OpType::QuantizeLinear
                    | OpType::QLinearConv
                    | OpType::QLinearMatMul
                    | OpType::QLinearAdd
                    | OpType::QLinearGlobalAveragePool
            );
            if is_quant_op {
                return has_quant;
            }
            // Only float outputs (for non-quantized ops)
            for out in &node.outputs {
                if !out.is_empty() {
                    if let Some(&dt) = type_map.get(out) {
                        if dt != DType::Float {
                            // Allow if quant_map knows about this tensor
                            let known_quant = has_quant
                                && quant_map
                                    .as_ref()
                                    .map(|qm| qm.tensor_quant.contains_key(out))
                                    .unwrap_or(false);
                            if !known_quant {
                                return false;
                            }
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
            if let Some(Some((op, _, _node))) = node_meta.get(i) {
                tracing::debug!(
                    "XNNPACK: non-eligible op at index {i}: {op}"
                );
            }
            i += 1;
        }
    }
    tracing::debug!("XNNPACK: {} runs, {} total ops", runs.len(), n);
    for run in &runs {
        let ops_in_run: Vec<_> = node_meta[run.clone()]
            .iter()
            .filter_map(|m| m.as_ref().map(|(op, _, _)| format!("{op}")))
            .collect();
        tracing::debug!("XNNPACK: run {:?} ({} ops): {:?}", run, ops_in_run.len(), &ops_in_run);
    }

    // Build a list of tensor names consumed by each node index.
    // node_meta has the inputs for real ops; auto-cast nodes (None in node_meta)
    // only consume initializers so they don't affect subgraph required_outputs.
    let consumed_by: Vec<Vec<String>> = (0..n)
        .map(|idx| {
            if let Some(Some((_, inputs, _))) = node_meta.get(idx) {
                inputs.clone()
            } else {
                vec![]
            }
        })
        .collect();

    let graph_output_set: std::collections::HashSet<&str> =
        graph_output_names.iter().map(|s| s.as_str()).collect();

    // Process runs in reverse order (to preserve indices)
    for run in runs.into_iter().rev() {
        // Collect all tensor names produced by this run
        let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();
        for idx in run.clone() {
            if let Some(Some((_, _, node))) = node_meta.get(idx) {
                for out in &node.outputs {
                    produced.insert(out.clone());
                }
            }
        }

        // A produced tensor is a required output if ANY node outside this run
        // consumes it, or if it's a graph output.
        let mut required_set: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Check nodes before the run
        for entry in consumed_by.iter().take(run.start) {
            for inp in entry {
                if !inp.is_empty() && produced.contains(inp) {
                    required_set.insert(inp.clone());
                }
            }
        }
        // Check nodes after the run
        for entry in consumed_by.iter().take(n).skip(run.end) {
            for inp in entry {
                if !inp.is_empty() && produced.contains(inp) {
                    required_set.insert(inp.clone());
                }
            }
        }
        // Check graph outputs
        for name in &produced {
            if graph_output_set.contains(name.as_str()) {
                required_set.insert(name.clone());
            }
        }

        let required_outputs: Vec<String> = required_set.into_iter().collect();

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

        // Build shape hints for XNNPACK from the layout-aware shape_map
        let shape_map_vec: HashMap<String, Vec<usize>> = shape_map
            .iter()
            .map(|(k, v)| (k.clone(), v.dims.to_vec()))
            .collect();

        // Remove the CPU plan nodes that are now covered by the XNNPACK subgraph
        nodes.drain(run.clone());

        let subgraph = if let Some(ref qm) = quant_map {
            xnnpack_subgraph::XnnpackSubgraph::with_quant_map(
                captured,
                required_outputs,
                shape_map_vec,
                qm.clone(),
            )
        } else {
            xnnpack_subgraph::XnnpackSubgraph::new(captured, required_outputs, shape_map_vec)
        };

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
    layout_only: &HashMap<String, Layout>,
) -> Layout {
    if op == OpType::LayoutTranspose {
        let perm = node.op.perm().unwrap_or(&[]);
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
        .and_then(|name| {
            shape_map
                .get(name)
                .map(|sl| sl.layout)
                .or_else(|| layout_only.get(name).copied())
        })
        .unwrap_or(Layout::NCHW)
}
