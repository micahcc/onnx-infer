use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::dims;
use crate::layers::Layer;
use crate::layers::OpType;
use crate::layers::abs;
use crate::layers::add;
use crate::layers::argmax;
use crate::layers::auto_cast;
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
use crate::layers::lstm;
use crate::layers::matmul;
use crate::layers::max_op;
use crate::layers::maxpool;
use crate::layers::min_op;
use crate::layers::mul;
use crate::layers::nms;
use crate::layers::nonzero;
use crate::layers::qlinear_add;
use crate::layers::qlinear_conv;
use crate::layers::qlinear_global_avg_pool;
use crate::layers::qlinear_matmul;
use crate::layers::quantize_linear;
use crate::layers::range;
use crate::layers::reduce_max;
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
use crate::onnx_ir::{Attr, Graph, Node};

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
    pub shape_map: HashMap<String, Dims>,
    pub type_map: HashMap<String, DType>,
    pub tensor_pool: HashMap<String, Tensor>,
}

impl Plan {
    pub fn build(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
    ) -> Result<Self> {
        Self::build_with_types(graph, input_sizes, &HashMap::new())
    }

    pub fn build_with_types(
        graph: &Graph,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
    ) -> Result<Self> {
        let initializers = graph.initializers.clone();

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

        let mut shape_map: HashMap<String, Dims> = HashMap::new();
        for (name, tensor) in &initializers {
            shape_map.insert(name.clone(), tensor.dims.clone());
        }
        let initializer_names: std::collections::HashSet<&str> =
            initializers.keys().map(|k| k.as_str()).collect();
        for input in &graph.inputs {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = &input.shape {
                if shape.iter().all(|&d| d > 0) || input_sizes.contains_key(&input.name) {
                    shape_map.insert(input.name.clone(), shape.clone());
                }
            }
        }
        for (name, user_dims) in input_sizes {
            if let Some(existing) = shape_map.get_mut(name) {
                if existing.len() == user_dims.len() {
                    for (i, d) in existing.iter_mut().enumerate() {
                        if *d == 0 {
                            *d = user_dims[i];
                        }
                    }
                } else {
                    *existing = user_dims.clone();
                }
            } else {
                shape_map.insert(name.clone(), user_dims.clone());
            }
        }

        let mut known_values: HashMap<String, Tensor> = HashMap::new();

        let mut nodes = Vec::new();
        let mut cast_counter = 0usize;
        #[cfg(feature = "xnnpack")]
        let mut node_meta: Vec<Option<(OpType, Vec<String>, Node)>> = Vec::new();

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

            if let Some(tensor) = try_propagate_value(
                op,
                node,
                &node.inputs,
                &known_values,
                &initializers,
                &shape_map,
            ) {
                if let Some(out_name) = out_name {
                    shape_map.insert(out_name.clone(), tensor.dims.clone());
                    known_values.insert(out_name.clone(), tensor);
                }
            } else if let Some(shape) =
                op.infer_output_shape(node, &node.inputs, &shape_map, &known_values)
            {
                if let Some(out_name) = out_name {
                    shape_map.insert(out_name.clone(), shape);
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
                if let Some(in_shape) = node
                    .inputs
                    .first()
                    .filter(|s| !s.is_empty())
                    .and_then(|n| shape_map.get(n))
                    .cloned()
                {
                    let axis_attr = node.attrs.get_int("axis").unwrap_or(0);
                    let rank = in_shape.len() as i64;
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
                        let mut out_shape = in_shape.clone();
                        out_shape[axis] = if let Some(ref sizes) = split_sizes {
                            sizes[i] as usize
                        } else {
                            let base = in_shape[axis] / num_outputs;
                            let rem = in_shape[axis] % num_outputs;
                            base + if i < rem { 1 } else { 0 }
                        };
                        shape_map.insert(out_name.clone(), out_shape);
                    }
                }
            }

            #[cfg(feature = "xnnpack")]
            node_meta.push(Some((op, modified_inputs.clone(), node.clone())));
            nodes.push(build_node(op, node, modified_inputs, &shape_map)?);
        }

        // XNNPACK subgraph compilation
        #[cfg(feature = "xnnpack")]
        {
            nodes =
                compile_xnnpack_subgraphs(nodes, node_meta, &shape_map, &type_map, &initializers)?;
        }

        // Pre-allocate tensors for all known shapes/types
        let mut tensor_pool: HashMap<String, Tensor> = HashMap::new();
        for (name, shape) in &shape_map {
            if initializers.contains_key(name) {
                continue;
            }
            let numel: usize = shape.iter().product();
            let dtype = type_map.get(name).copied().unwrap_or(DType::Float);
            let tensor = match dtype {
                DType::Float => Tensor::new(shape.clone(), vec![0.0; numel]),
                DType::Int64 => Tensor::new_i64(shape.clone(), vec![0; numel]),
                DType::String => Tensor::new_strings(shape.clone(), vec![vec![]; numel]),
            };
            tensor_pool.insert(name.clone(), tensor);
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
    shape_map: &HashMap<String, Dims>,
) -> Option<Tensor> {
    if op == OpType::Shape {
        let name = input_names.first().filter(|s| !s.is_empty())?;
        let shape = shape_map.get(name)?;
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Some(Tensor::new_i64(dims![dims.len()], dims));
    }

    if op == OpType::Constant {
        return match node.attrs.get("value") {
            Some(Attr::Tensor(t)) => Some(t.clone()),
            _ => None,
        };
    }

    match op {
        OpType::Gather
        | OpType::Unsqueeze
        | OpType::Squeeze
        | OpType::Concat
        | OpType::Cast
        | OpType::Identity
        | OpType::Reshape
        | OpType::Flatten
        | OpType::Slice => {}
        _ => return None,
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
    shape_map: &HashMap<String, Dims>,
) -> Result<PlanNode> {
    if op == OpType::Loop {
        let body = match node.attrs.get("body") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => return Err(InferenceError::InvalidModel("Loop: no body graph".into())),
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
            _ => return Err(InferenceError::InvalidModel("If: no then_branch".into())),
        };
        let else_branch = match node.attrs.get("else_branch") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => return Err(InferenceError::InvalidModel("If: no else_branch".into())),
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
            _ => return Err(InferenceError::InvalidModel("Scan: no body graph".into())),
        };
        let num_scan_inputs = node.attrs.get_int("num_scan_inputs").unwrap_or(0) as usize;
        let scan_input_directions = node.attrs.get_ints("scan_input_directions").unwrap_or_default();
        let scan_output_directions = node.attrs.get_ints("scan_output_directions").unwrap_or_default();
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
            if let Some(s) = shape_map.get(name) {
                input_shapes[i] = s.as_slice();
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
        OpType::Tanh => Box::new(tanh::Tanh::new(inputs)),
        OpType::Expand => Box::new(expand::Expand::new(inputs)),
        OpType::Less => Box::new(less::Less::new(inputs)),
        OpType::Equal => Box::new(equal::Equal::new(inputs)),
        OpType::Greater => Box::new(greater::Greater::new(inputs)),
        OpType::Max => Box::new(max_op::Max::new(inputs)),
        OpType::Min => Box::new(min_op::Min::new(inputs)),
        OpType::NonZero => Box::new(nonzero::NonZero::new(inputs)),
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
            let mode = node.attrs.get_string("mode").unwrap_or_else(|| "avg".to_string());
            let oh = node.attrs.get_int("output_height").unwrap_or(1) as usize;
            let ow = node.attrs.get_int("output_width").unwrap_or(1) as usize;
            let sr = node.attrs.get_int("sampling_ratio").unwrap_or(0) as usize;
            let ss = node.attrs.get_float("spatial_scale").unwrap_or(1.0);
            Box::new(roi_align::RoiAlign::new(inputs, mode, oh, ow, sr, ss))
        }
        OpType::ConstantOfShape => {
            let (fill_f32, fill_i64, dtype) = match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => match t.dtype() {
                    DType::Int64 => (0.0, t.ints().first().copied().unwrap_or(0), DType::Int64),
                    _ => (t.floats().first().copied().unwrap_or(0.0), 0, DType::Float),
                },
                _ => (0.0, 0, DType::Float),
            };
            Box::new(constant_of_shape::ConstantOfShape::new(
                inputs, fill_f32, fill_i64, dtype,
            ))
        }
        OpType::Ceil => Box::new(ceil::Ceil::new(inputs)),
        OpType::Round => Box::new(round::Round::new(inputs)),
        OpType::Softmax => Box::new(softmax::Softmax::new(
            inputs,
            node.attrs.get_int("axis").unwrap_or(-1),
            input_shapes[0],
        )),
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
            let pa = node.attrs.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = node.attrs.get_ints("dilations").unwrap_or_else(|| vec![1, 1]);
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
            let pa = node.attrs.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
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
        OpType::Transpose => {
            let perm = node.attrs.get_ints("perm").map(|p| p.iter().map(|&v| v as usize).collect());
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
            let ct = node.attrs.get_string("coordinate_transformation_mode").unwrap_or_default();
            let nm = node.attrs.get_string("nearest_mode").unwrap_or_default();
            Box::new(resize::Resize::new(inputs, &ct, &nm))
        }
        OpType::Upsample => {
            let mode = node.attrs.get_string("mode").unwrap_or_else(|| "nearest".to_string());
            let nm = if mode == "nearest" { "floor" } else { "" };
            let resize_inputs = vec![
                inputs[0].clone(),
                String::new(),
                inputs[1].clone(),
            ];
            Box::new(resize::Resize::new(resize_inputs, "asymmetric", nm))
        }
        OpType::Reshape => Box::new(reshape::Reshape::new(
            inputs,
            node.attrs.get_ints("shape"),
        )),
        OpType::Constant => {
            let tensor = match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => t.clone(),
                _ => {
                    return Err(InferenceError::InvalidModel(
                        "Constant has no value".into(),
                    ))
                }
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
            let pa = node.attrs.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = node.attrs.get_ints("dilations").unwrap_or_else(|| vec![1, 1]);
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
            _ => return Err(InferenceError::InvalidModel("Loop: no body graph".into())),
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
            _ => return Err(InferenceError::InvalidModel("If: no then_branch".into())),
        };
        let else_branch = match node.attrs.get("else_branch") {
            Some(Attr::Graph(g)) => (**g).clone(),
            _ => return Err(InferenceError::InvalidModel("If: no else_branch".into())),
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
            _ => return Err(InferenceError::InvalidModel("Scan: no body graph".into())),
        };
        let num_scan_inputs = node.attrs.get_int("num_scan_inputs").unwrap_or(0) as usize;
        let scan_input_directions = node.attrs.get_ints("scan_input_directions").unwrap_or_default();
        let scan_output_directions = node.attrs.get_ints("scan_output_directions").unwrap_or_default();
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
        casted.copy_cast_f32(src);
        values.insert(cast_name.clone(), casted);
        modified_inputs[i] = cast_name;
    }

    let exec_shape_map: HashMap<String, Dims> = values
        .iter()
        .map(|(k, v)| (k.clone(), v.dims.clone()))
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
        PlanNode::XnnpackSubgraph(_) => Err(InferenceError::InvalidModel(
            "XnnpackSubgraph cannot be executed via execute_node".into(),
        )),
    }
}

#[cfg(feature = "xnnpack")]
fn compile_xnnpack_subgraphs(
    mut nodes: Vec<PlanNode>,
    node_meta: Vec<Option<(OpType, Vec<String>, Node)>>,
    shape_map: &HashMap<String, Dims>,
    type_map: &HashMap<String, DType>,
    initializers: &HashMap<String, Tensor>,
) -> Result<Vec<PlanNode>> {
    use super::xnnpack_subgraph::{CapturedOp, is_xnnpack_compatible};

    assert_eq!(nodes.len(), node_meta.len());

    let compatible: Vec<bool> = node_meta
        .iter()
        .map(|meta| {
            let Some((op, inputs, node)) = meta else {
                return false;
            };
            if !is_xnnpack_compatible(*op) {
                return false;
            }
            for out in &node.outputs {
                if !out.is_empty() {
                    if let Some(&dt) = type_map.get(out) {
                        if dt != DType::Float {
                            return false;
                        }
                    }
                }
            }
            for inp in inputs {
                if !inp.is_empty() && !initializers.contains_key(inp) {
                    if let Some(&dt) = type_map.get(inp) {
                        if dt != DType::Float {
                            return false;
                        }
                    }
                }
            }
            let needs_4d = matches!(
                op,
                OpType::Conv | OpType::MaxPool | OpType::GlobalAveragePool
            );
            if needs_4d {
                let first_input = inputs.first().filter(|s| !s.is_empty());
                if let Some(name) = first_input {
                    if let Some(s) = shape_map.get(name) {
                        if s.len() != 4 {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
            if *op == OpType::MaxPool {
                let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();
                if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                    return false;
                }
                let pads = node.attrs.get_ints("pads").unwrap_or_default();
                if pads.len() >= 4 && (pads[0] != pads[2] || pads[1] != pads[3]) {
                    return false;
                }
            }
            if matches!(op, OpType::Conv | OpType::Gemm) {
                if inputs.len() < 2 || !initializers.contains_key(&inputs[1]) {
                    return false;
                }
            }
            if *op == OpType::MatMul && inputs.len() >= 2 && !initializers.contains_key(&inputs[1])
            {
                let a = inputs.first().and_then(|n| shape_map.get(n));
                let b = inputs.get(1).and_then(|n| shape_map.get(n));
                if a.is_none() || b.is_none() {
                    return false;
                }
            }
            if *op == OpType::BatchNormalization {
                if inputs.len() < 5 {
                    return false;
                }
                for i in 1..5 {
                    if !initializers.contains_key(&inputs[i]) {
                        return false;
                    }
                }
            }
            if *op == OpType::Concat {
                for name in inputs.iter().chain(node.outputs.iter()) {
                    if name.is_empty() {
                        continue;
                    }
                    if let Some(s) = shape_map.get(name) {
                        if s.len() != 4 {
                            return false;
                        }
                    }
                }
            }
            true
        })
        .collect();

    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut i = 0;
    while i < compatible.len() {
        if compatible[i] {
            let start = i;
            while i < compatible.len() && compatible[i] {
                i += 1;
            }
            if i - start >= 2 {
                runs.push(start..i);
            }
        } else {
            i += 1;
        }
    }

    if runs.is_empty() {
        return Ok(nodes);
    }

    let shape_map_vec: HashMap<String, Vec<usize>> = shape_map
        .iter()
        .map(|(k, v)| (k.clone(), v.to_vec()))
        .collect();

    let all_node_inputs: Vec<std::collections::HashSet<&str>> = node_meta
        .iter()
        .map(|meta| {
            if let Some((_, inputs, _)) = meta {
                inputs.iter().map(|s| s.as_str()).collect()
            } else {
                std::collections::HashSet::new()
            }
        })
        .collect();

    for run in runs.into_iter().rev() {
        let mut run_outputs: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for meta in &node_meta[run.clone()] {
            if let Some((_, _, node)) = meta {
                for out in &node.outputs {
                    if !out.is_empty() {
                        run_outputs.insert(out.as_str());
                    }
                }
            }
        }

        let mut required_output_set: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for i in 0..node_meta.len() {
            if run.contains(&i) {
                continue;
            }
            for name in &all_node_inputs[i] {
                if run_outputs.contains(name) {
                    required_output_set.insert(name.to_string());
                }
            }
        }
        if let Some(meta) = node_meta[run.end - 1].as_ref() {
            for out in &meta.2.outputs {
                if !out.is_empty() {
                    required_output_set.insert(out.clone());
                }
            }
        }
        let required_outputs: Vec<String> = required_output_set.into_iter().collect();

        let captured: Vec<CapturedOp> = node_meta[run.clone()]
            .iter()
            .map(|meta| {
                let (op, inputs, node) = meta.as_ref().unwrap();
                CapturedOp {
                    op: *op,
                    inputs: inputs.clone(),
                    outputs: node.outputs.clone(),
                    node: node.clone(),
                }
            })
            .collect();

        let op_names: Vec<&str> = captured
            .iter()
            .map(|c| match c.op {
                OpType::Conv => "Conv",
                OpType::Relu => "Relu",
                OpType::Add => "Add",
                OpType::MaxPool => "MaxPool",
                OpType::GlobalAveragePool => "GlobalAvgPool",
                OpType::Flatten => "Flatten",
                OpType::Gemm => "Gemm",
                OpType::Softmax => "Softmax",
                OpType::BatchNormalization => "BatchNorm",
                _ => "other",
            })
            .collect();
        tracing::info!(
            "XNNPACK: planning subgraph of {} ops: {:?}",
            captured.len(),
            op_names
        );

        let mut sub_initializers: HashMap<String, Tensor> = HashMap::new();
        for cap in &captured {
            for inp in &cap.inputs {
                if !inp.is_empty() && initializers.contains_key(inp) {
                    if let Some(t) = initializers.get(inp) {
                        sub_initializers.insert(inp.clone(), t.clone());
                    }
                }
            }
        }

        let mut subgraph = super::xnnpack_subgraph::XnnpackSubgraph::new(
            captured,
            required_outputs,
            shape_map_vec.clone(),
            sub_initializers,
        );

        let removed: Vec<_> = nodes.drain(run.clone()).collect();
        subgraph.fallback_nodes = removed;
        nodes.insert(run.start, PlanNode::XnnpackSubgraph(Box::new(subgraph)));
    }

    Ok(nodes)
}
