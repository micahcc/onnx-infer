use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::InferenceError;
use crate::ONNX_INT32;
use crate::ONNX_INT64;
use crate::ONNX_STRING;
use crate::Result;
use crate::Tensor;
use crate::dims;
use crate::get_attr_float;
use crate::get_attr_int;
use crate::get_attr_ints;
use crate::get_attr_string;
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
use crate::onnx::NodeProto;

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
        graph: &crate::onnx::GraphProto,
        input_sizes: &HashMap<String, Dims>,
    ) -> Result<Self> {
        Self::build_with_types(graph, input_sizes, &HashMap::new())
    }

    /// Extract input shapes from the graph's input type information.
    /// Only includes non-initializer inputs with fully concrete dimensions.
    pub fn infer_input_sizes(graph: &crate::onnx::GraphProto) -> HashMap<String, Dims> {
        let initializer_names: std::collections::HashSet<&str> =
            graph.initializer.iter().map(|i| i.name.as_str()).collect();
        let mut sizes = HashMap::new();
        for input in &graph.input {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = Self::extract_shape(input) {
                sizes.insert(input.name.clone(), shape);
            }
        }
        sizes
    }

    fn extract_shape(input: &crate::onnx::ValueInfoProto) -> Option<Dims> {
        let tt = match input.r#type.as_ref()?.value.as_ref()? {
            crate::onnx::type_proto::Value::TensorType(tt) => tt,
            _ => return None,
        };
        let shape_proto = tt.shape.as_ref()?;
        let mut dims = Dims::new();
        for d in &shape_proto.dim {
            match &d.value {
                Some(crate::onnx::tensor_shape_proto::dimension::Value::DimValue(v)) if *v > 0 => {
                    dims.push(*v as usize);
                }
                _ => return None, // symbolic or missing dimension
            }
        }
        if dims.is_empty() {
            return None;
        }
        Some(dims)
    }

    /// Extract shape from graph input, using 0 for dynamic/symbolic dimensions.
    pub fn extract_shape_partial(input: &crate::onnx::ValueInfoProto) -> Option<Dims> {
        let tt = match input.r#type.as_ref()?.value.as_ref()? {
            crate::onnx::type_proto::Value::TensorType(tt) => tt,
            _ => return None,
        };
        let shape_proto = tt.shape.as_ref()?;
        if shape_proto.dim.is_empty() {
            return None;
        }
        let mut dims = Dims::new();
        for d in &shape_proto.dim {
            match &d.value {
                Some(crate::onnx::tensor_shape_proto::dimension::Value::DimValue(v)) if *v > 0 => {
                    dims.push(*v as usize);
                }
                _ => dims.push(0), // symbolic or missing → 0
            }
        }
        Some(dims)
    }

    pub fn build_with_types(
        graph: &crate::onnx::GraphProto,
        input_sizes: &HashMap<String, Dims>,
        type_hints: &HashMap<String, DType>,
    ) -> Result<Self> {
        let mut initializers = HashMap::new();
        for init in &graph.initializer {
            if !init.name.is_empty() {
                initializers.insert(init.name.clone(), Tensor::from_proto(init)?);
            }
        }

        let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

        let mut type_map: HashMap<String, DType> = HashMap::new();
        for (name, tensor) in &initializers {
            type_map.insert(name.clone(), tensor.dtype());
        }
        for (name, &dtype) in type_hints {
            type_map.insert(name.clone(), dtype);
        }
        for input in &graph.input {
            if !type_map.contains_key(&input.name) {
                let dtype = input
                    .r#type
                    .as_ref()
                    .and_then(|t| t.value.as_ref())
                    .map(|v| match v {
                        crate::onnx::type_proto::Value::TensorType(tt) => {
                            if tt.elem_type == ONNX_INT32 || tt.elem_type == ONNX_INT64 {
                                DType::Int64
                            } else if tt.elem_type == ONNX_STRING {
                                DType::String
                            } else {
                                DType::Float
                            }
                        }
                        _ => DType::Float,
                    })
                    .unwrap_or(DType::Float);
                type_map.insert(input.name.clone(), dtype);
            }
        }

        let mut shape_map: HashMap<String, Dims> = HashMap::new();
        for (name, tensor) in &initializers {
            shape_map.insert(name.clone(), tensor.dims.clone());
        }
        // Merge graph-inferred input sizes first, then override with explicit ones.
        // Use partial shapes (with 0 for dynamic dims) so that user-provided
        // input_sizes can selectively fill in just the dynamic dimensions.
        let initializer_names: std::collections::HashSet<&str> =
            graph.initializer.iter().map(|i| i.name.as_str()).collect();
        for input in &graph.input {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = Self::extract_shape_partial(input) {
                // Only insert if fully concrete (no zeros) OR user will override
                if shape.iter().all(|&d| d > 0) || input_sizes.contains_key(&input.name) {
                    shape_map.insert(input.name.clone(), shape);
                }
            }
        }
        // User-provided sizes: merge dimension-by-dimension, replacing 0s
        for (name, user_dims) in input_sizes {
            if let Some(existing) = shape_map.get_mut(name) {
                if existing.len() == user_dims.len() {
                    // Fill in zeros from user-provided dims
                    for (i, d) in existing.iter_mut().enumerate() {
                        if *d == 0 {
                            *d = user_dims[i];
                        }
                    }
                } else {
                    // Different rank — use user dims entirely
                    *existing = user_dims.clone();
                }
            } else {
                shape_map.insert(name.clone(), user_dims.clone());
            }
        }

        let mut known_values: HashMap<String, Tensor> = HashMap::new();

        let mut nodes = Vec::new();
        let mut cast_counter = 0usize;
        // Parallel metadata for XNNPACK subgraph identification
        #[cfg(feature = "xnnpack")]
        let mut node_meta: Vec<Option<(OpType, Vec<String>, crate::onnx::NodeProto)>> = Vec::new();

        for node in &graph.node {
            let op = OpType::parse(&node.op_type).map_err(InferenceError::UnsupportedOperator)?;

            let expected = op.expected_input_dtypes();
            let mut modified_inputs = node.input.clone();

            let mut input_types = Vec::new();
            for name in &node.input {
                if let Some(&dt) = type_map.get(name) {
                    input_types.push(dt);
                }
            }

            for (i, input_name) in node.input.iter().enumerate() {
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
                            node_meta.push(None); // auto-cast nodes are not XNNPACK-compatible
                            type_map.insert(cast_name.clone(), DType::Float);
                            modified_inputs[i] = cast_name;
                        }
                    }
                }
            }

            let out_dtype = op.infer_output_dtype(node, &input_types);
            let out_name = node.output.first().filter(|s| !s.is_empty());
            if let Some(out_name) = out_name {
                type_map.insert(out_name.clone(), out_dtype);
            }

            if let Some(tensor) = try_propagate_value(
                op,
                node,
                &node.input,
                &known_values,
                &initializers,
                &shape_map,
            ) {
                if let Some(out_name) = out_name {
                    shape_map.insert(out_name.clone(), tensor.dims.clone());
                    known_values.insert(out_name.clone(), tensor);
                }
            } else if let Some(shape) =
                op.infer_output_shape(node, &node.input, &shape_map, &known_values)
            {
                if let Some(out_name) = out_name {
                    shape_map.insert(out_name.clone(), shape);
                }
            }

            // For Split, infer types and shapes for all outputs
            if op == OpType::Split {
                let in_dtype = input_types.first().copied().unwrap_or(DType::Float);
                for out_name in &node.output {
                    if !out_name.is_empty() {
                        type_map.insert(out_name.clone(), in_dtype);
                    }
                }
                if let Some(in_shape) = node
                    .input
                    .first()
                    .filter(|s| !s.is_empty())
                    .and_then(|n| shape_map.get(n))
                    .cloned()
                {
                    let axis_attr = get_attr_int(node, "axis").unwrap_or(0);
                    let rank = in_shape.len() as i64;
                    let axis = if axis_attr < 0 {
                        (rank + axis_attr) as usize
                    } else {
                        axis_attr as usize
                    };
                    let split_sizes = get_attr_ints(node, "split");
                    let num_outputs = node.output.len();
                    for (i, out_name) in node.output.iter().enumerate() {
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

        // XNNPACK subgraph compilation: find runs of compatible ops and compile them
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
    node: &NodeProto,
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
        let attr = node.attribute.iter().find(|a| a.name == "value")?;
        return Tensor::from_proto(attr.t.as_ref()?).ok();
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
    node: &NodeProto,
    inputs: Vec<String>,
    shape_map: &HashMap<String, Dims>,
) -> Result<PlanNode> {
    if op == OpType::Loop {
        let body = node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("Loop: no body graph".into()))?
            .clone();
        return Ok(PlanNode::Loop(Box::new(loop_op::Loop::new(
            inputs,
            node.output.clone(),
            body,
        ))));
    }

    if op == OpType::Split {
        let axis = get_attr_int(node, "axis").unwrap_or(0);
        let split_sizes = get_attr_ints(node, "split").unwrap_or_default();
        return Ok(PlanNode::Split(Box::new(split::Split::new(
            inputs,
            node.output.clone(),
            axis,
            split_sizes,
        ))));
    }

    if op == OpType::If {
        let then_branch = node
            .attribute
            .iter()
            .find(|a| a.name == "then_branch")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("If: no then_branch".into()))?
            .clone();
        let else_branch = node
            .attribute
            .iter()
            .find(|a| a.name == "else_branch")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("If: no else_branch".into()))?
            .clone();
        return Ok(PlanNode::If(Box::new(if_op::If::new(
            inputs,
            node.output.clone(),
            then_branch,
            else_branch,
        ))));
    }

    if op == OpType::TopK {
        let axis = get_attr_int(node, "axis").unwrap_or(-1);
        let largest = get_attr_int(node, "largest").unwrap_or(1) != 0;
        return Ok(PlanNode::TopK(Box::new(topk::TopK::new(
            inputs,
            node.output.clone(),
            axis,
            largest,
        ))));
    }

    if op == OpType::Scan {
        let body = node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("Scan: no body graph".into()))?
            .clone();
        let num_scan_inputs = get_attr_int(node, "num_scan_inputs").unwrap_or(0) as usize;
        let scan_input_directions =
            get_attr_ints(node, "scan_input_directions").unwrap_or_default();
        let scan_output_directions =
            get_attr_ints(node, "scan_output_directions").unwrap_or_default();
        return Ok(PlanNode::Scan(Box::new(scan::Scan::new(
            inputs,
            node.output.clone(),
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
        ))));
    }

    let output = if node.output.is_empty() || node.output[0].is_empty() {
        String::new()
    } else {
        node.output[0].clone()
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
            get_attr_float(node, "alpha").unwrap_or(0.01),
        )),
        OpType::Clip => Box::new(clip::Clip::new(
            inputs,
            get_attr_float(node, "min").unwrap_or(f32::NEG_INFINITY),
            get_attr_float(node, "max").unwrap_or(f32::INFINITY),
        )),
        OpType::BatchNormalization => Box::new(batch_norm::BatchNorm::new(
            inputs,
            get_attr_float(node, "epsilon").unwrap_or(1e-5),
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
            get_attr_int(node, "axis").unwrap_or(0),
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::RoiAlign => {
            let mode = get_attr_string(node, "mode").unwrap_or_else(|| "avg".to_string());
            let oh = get_attr_int(node, "output_height").unwrap_or(1) as usize;
            let ow = get_attr_int(node, "output_width").unwrap_or(1) as usize;
            let sr = get_attr_int(node, "sampling_ratio").unwrap_or(0) as usize;
            let ss = get_attr_float(node, "spatial_scale").unwrap_or(1.0);
            Box::new(roi_align::RoiAlign::new(inputs, mode, oh, ow, sr, ss))
        }
        OpType::ConstantOfShape => {
            let (fill_f32, fill_i64, dtype) = node
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
                .map(|t| {
                    if t.data_type == ONNX_INT32 || t.data_type == ONNX_INT64 {
                        let v = if !t.int64_data.is_empty() {
                            t.int64_data[0]
                        } else if !t.raw_data.is_empty() && t.data_type == ONNX_INT64 {
                            i64::from_le_bytes(t.raw_data[..8].try_into().unwrap_or([0; 8]))
                        } else if !t.raw_data.is_empty() && t.data_type == ONNX_INT32 {
                            i32::from_le_bytes(t.raw_data[..4].try_into().unwrap_or([0; 4])) as i64
                        } else {
                            0
                        };
                        (0.0, v, DType::Int64)
                    } else {
                        let v = if !t.float_data.is_empty() {
                            t.float_data[0]
                        } else if !t.raw_data.is_empty() {
                            f32::from_le_bytes(t.raw_data[..4].try_into().unwrap_or([0; 4]))
                        } else {
                            0.0
                        };
                        (v, 0, DType::Float)
                    }
                })
                .unwrap_or((0.0, 0, DType::Float));
            Box::new(constant_of_shape::ConstantOfShape::new(
                inputs, fill_f32, fill_i64, dtype,
            ))
        }
        OpType::Ceil => Box::new(ceil::Ceil::new(inputs)),
        OpType::Round => Box::new(round::Round::new(inputs)),
        OpType::Softmax => Box::new(softmax::Softmax::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(-1),
            input_shapes[0],
        )),
        OpType::Softplus => Box::new(softplus::Softplus::new(inputs)),
        OpType::Add => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(add::Add::new(inputs, lb, axis))
        }
        OpType::Sub => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(sub::Sub::new(inputs, lb, axis))
        }
        OpType::Mul => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(mul::Mul::new(inputs, lb, axis))
        }
        OpType::Div => {
            let lb = get_attr_int(node, "broadcast").unwrap_or(0) != 0;
            let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
            Box::new(div::Div::new(inputs, lb, axis))
        }
        OpType::Conv => {
            let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
            let st = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pa = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let di = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
            let gr = get_attr_int(node, "group").unwrap_or(1) as usize;
            let ap = get_attr_string(node, "auto_pad").unwrap_or_default();
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
            get_attr_float(node, "alpha").unwrap_or(1.0),
            get_attr_float(node, "beta").unwrap_or(1.0),
            get_attr_int(node, "transA").unwrap_or(0) != 0,
            get_attr_int(node, "transB").unwrap_or(0) != 0,
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::MaxPool => {
            let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
            let st = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pa = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let ap = get_attr_string(node, "auto_pad").unwrap_or_default();
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
            get_attr_int(node, "axis").unwrap_or(1) as usize,
            input_shapes[0],
        )),
        OpType::Shape => Box::new(shape_op::Shape::new(inputs)),
        OpType::Gather => Box::new(gather::Gather::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(0),
            input_shapes[0],
            input_shapes[1],
        )),
        OpType::Unsqueeze => Box::new(unsqueeze::Unsqueeze::new(inputs, node)),
        OpType::Concat => Box::new(concat::Concat::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(0),
            &input_shapes,
        )),
        OpType::Identity => Box::new(identity::Identity::new(inputs)),
        OpType::Cast => Box::new(cast::Cast::new(
            inputs,
            get_attr_int(node, "to").unwrap_or(1),
        )),
        OpType::Transpose => {
            let perm = get_attr_ints(node, "perm").map(|p| p.iter().map(|&v| v as usize).collect());
            Box::new(transpose::Transpose::new(inputs, perm, input_shapes[0]))
        }
        OpType::Squeeze => Box::new(squeeze::Squeeze::new(inputs, node)),
        OpType::Slice => {
            if let Some(attr_starts) = get_attr_ints(node, "starts") {
                let attr_ends = get_attr_ints(node, "ends").unwrap_or_default();
                let attr_axes = get_attr_ints(node, "axes");
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
            let ct = get_attr_string(node, "coordinate_transformation_mode").unwrap_or_default();
            let nm = get_attr_string(node, "nearest_mode").unwrap_or_default();
            Box::new(resize::Resize::new(inputs, &ct, &nm))
        }
        OpType::Upsample => {
            // Upsample (opset 7-9): inputs are [X, scales].
            // Remap to Resize layout [X, roi(empty), scales].
            let mode = get_attr_string(node, "mode").unwrap_or_else(|| "nearest".to_string());
            let nm = if mode == "nearest" { "floor" } else { "" };
            let resize_inputs = vec![
                inputs[0].clone(),
                String::new(), // empty roi
                inputs[1].clone(),
            ];
            Box::new(resize::Resize::new(resize_inputs, "asymmetric", nm))
        }
        OpType::Reshape => Box::new(reshape::Reshape::new(inputs, node)),
        OpType::Constant => {
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
        OpType::ReduceMin => Box::new(reduce_min::ReduceMin::new(
            inputs,
            get_attr_int(node, "keepdims").unwrap_or(1) != 0,
            node,
            input_shapes[0],
        )),
        OpType::NonMaxSuppression => Box::new(nms::Nms::new(inputs)),
        OpType::QuantizeLinear => Box::new(quantize_linear::QuantizeLinear::new(inputs)),
        OpType::DequantizeLinear => Box::new(dequantize_linear::DequantizeLinear::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(1),
            input_shapes[0],
        )),
        OpType::QLinearConv => {
            let has_bias = inputs.len() > 8 && !inputs[8].is_empty();
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
            // QLinearConv: quantized x is inputs[0], quantized w is inputs[3]
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
            // QLinearMatMul: quantized a is inputs[0], quantized b is inputs[3]
            let inner = matmul::MatMul::new(
                vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()],
                input_shapes[0],
                input_shapes[3],
            );
            Box::new(qlinear_matmul::QLinearMatMul::new(inputs, inner))
        }
        OpType::QLinearGlobalAveragePool => {
            // QLinearGlobalAvgPool: quantized x is inputs[0]
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
            get_attr_int(node, "axis").unwrap_or(0),
            get_attr_int(node, "keepdims").unwrap_or(1) != 0,
            get_attr_int(node, "select_last_index").unwrap_or(0) != 0,
        )),
        OpType::CategoryMapper => {
            let cats_strings: Vec<Vec<u8>> = node
                .attribute
                .iter()
                .find(|a| a.name == "cats_strings")
                .map(|a| a.strings.iter().map(|s| s.to_vec()).collect())
                .unwrap_or_default();
            let cats_int64s: Vec<i64> = node
                .attribute
                .iter()
                .find(|a| a.name == "cats_int64s")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            let default_int64 = get_attr_int(node, "default_int64").unwrap_or(-1);
            Box::new(category_mapper::CategoryMapper::new(
                inputs,
                cats_strings,
                cats_int64s,
                default_int64,
            ))
        }
        OpType::Compress => {
            let axis = get_attr_int(node, "axis");
            Box::new(compress::Compress::new(inputs, axis))
        }
        OpType::Dropout => Box::new(dropout::Dropout::new(inputs)),
        OpType::Hardmax => Box::new(hardmax::Hardmax::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(-1),
        )),
        OpType::Lstm => {
            let hs = get_attr_int(node, "hidden_size").unwrap_or(1) as usize;
            let dir_str = get_attr_string(node, "direction").unwrap_or_default();
            let direction = match dir_str.as_str() {
                "reverse" => lstm::LstmDirection::Reverse,
                "bidirectional" => lstm::LstmDirection::Bidirectional,
                _ => lstm::LstmDirection::Forward,
            };
            Box::new(lstm::Lstm::new(inputs, node.output.clone(), hs, direction))
        }
        OpType::ReduceMax => Box::new(reduce_max::ReduceMax::new(
            inputs,
            get_attr_int(node, "keepdims").unwrap_or(1) != 0,
            node,
        )),
        OpType::ReduceSum => Box::new(reduce_sum::ReduceSum::new(
            inputs,
            get_attr_int(node, "keepdims").unwrap_or(1) != 0,
            node,
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
            get_attr_float(node, "alpha").unwrap_or(1.0),
        )),
        OpType::Celu => Box::new(unary_ops::Celu::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(1.0),
        )),
        OpType::Selu => Box::new(unary_ops::Selu::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(1.673_263_2),
            get_attr_float(node, "gamma").unwrap_or(1.050_701),
        )),
        OpType::HardSigmoid => Box::new(unary_ops::HardSigmoid::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(0.2),
            get_attr_float(node, "beta").unwrap_or(0.5),
        )),
        OpType::ThresholdedRelu => Box::new(unary_ops::ThresholdedRelu::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(1.0),
        )),
        OpType::Loop | OpType::Split | OpType::If | OpType::TopK | OpType::Scan => {
            unreachable!()
        }
    };

    Ok(PlanNode::Single { output, layer })
}

pub fn execute_node(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let op = OpType::parse(&node.op_type).map_err(InferenceError::UnsupportedOperator)?;

    let _span = tracing::trace_span!("op", op = %op, name = %node.name).entered();

    if op == OpType::Loop {
        let body = node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("Loop: no body graph".into()))?
            .clone();
        let mut loop_layer = loop_op::Loop::new(node.input.clone(), node.output.clone(), body);
        return loop_layer.execute(values);
    }

    if op == OpType::Split {
        let axis = get_attr_int(node, "axis").unwrap_or(0);
        let split_sizes = get_attr_ints(node, "split").unwrap_or_default();
        let mut split_layer =
            split::Split::new(node.input.clone(), node.output.clone(), axis, split_sizes);
        return split_layer.execute(values);
    }

    if op == OpType::If {
        let then_branch = node
            .attribute
            .iter()
            .find(|a| a.name == "then_branch")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("If: no then_branch".into()))?
            .clone();
        let else_branch = node
            .attribute
            .iter()
            .find(|a| a.name == "else_branch")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("If: no else_branch".into()))?
            .clone();
        let mut if_layer = if_op::If::new(
            node.input.clone(),
            node.output.clone(),
            then_branch,
            else_branch,
        );
        return if_layer.execute(values);
    }

    if op == OpType::TopK {
        let axis = get_attr_int(node, "axis").unwrap_or(-1);
        let largest = get_attr_int(node, "largest").unwrap_or(1) != 0;
        let mut topk_layer =
            topk::TopK::new(node.input.clone(), node.output.clone(), axis, largest);
        return topk_layer.execute(values);
    }

    if op == OpType::Scan {
        let body = node
            .attribute
            .iter()
            .find(|a| a.name == "body")
            .and_then(|a| a.g.as_ref())
            .ok_or_else(|| InferenceError::InvalidModel("Scan: no body graph".into()))?
            .clone();
        let num_scan_inputs = get_attr_int(node, "num_scan_inputs").unwrap_or(0) as usize;
        let scan_input_directions =
            get_attr_ints(node, "scan_input_directions").unwrap_or_default();
        let scan_output_directions =
            get_attr_ints(node, "scan_output_directions").unwrap_or_default();
        let mut scan_layer = scan::Scan::new(
            node.input.clone(),
            node.output.clone(),
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
        );
        return scan_layer.execute(values);
    }

    if node.output.is_empty() || node.output[0].is_empty() {
        return Ok(());
    }

    let expected = op.expected_input_dtypes();
    let mut to_cast: Vec<(usize, String)> = Vec::new();
    for (i, input_name) in node.input.iter().enumerate() {
        if input_name.is_empty() {
            continue;
        }
        if let Some(Some(expected_dt)) = expected.get(i) {
            if let Some(tensor) = values.get(input_name) {
                if tensor.dtype() != *expected_dt {
                    {
                        to_cast.push((i, input_name.clone()));
                    }
                }
            }
        }
    }

    let mut modified_inputs = node.input.clone();
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

/// Identify runs of XNNPACK-compatible ops and compile them into subgraph nodes.
///
/// A "run" is a maximal contiguous sequence of ops where:
/// - The op is XNNPACK-compatible
/// - All non-initializer inputs either come from within the run or from outside
/// - The shapes are statically known (required for XNNPACK subgraph)
/// - The op works on float data
///
/// Runs shorter than 2 nodes are not worth compiling.
#[cfg(feature = "xnnpack")]
fn compile_xnnpack_subgraphs(
    mut nodes: Vec<PlanNode>,
    node_meta: Vec<Option<(OpType, Vec<String>, crate::onnx::NodeProto)>>,
    shape_map: &HashMap<String, Dims>,
    type_map: &HashMap<String, DType>,
    initializers: &HashMap<String, Tensor>,
) -> Result<Vec<PlanNode>> {
    use super::xnnpack_subgraph::{CapturedOp, is_xnnpack_compatible};

    assert_eq!(nodes.len(), node_meta.len());

    // Identify which nodes are XNNPACK-compatible single nodes with known shapes
    let compatible: Vec<bool> = node_meta
        .iter()
        .map(|meta| {
            let Some((op, inputs, node)) = meta else {
                return false;
            };
            if !is_xnnpack_compatible(*op) {
                return false;
            }
            // XNNPACK only handles Float — reject ops with non-Float outputs
            for out in &node.output {
                if !out.is_empty() {
                    if let Some(&dt) = type_map.get(out) {
                        if dt != DType::Float {
                            return false;
                        }
                    }
                }
            }
            // Also reject ops where non-initializer inputs are non-Float
            for inp in inputs {
                if !inp.is_empty() && !initializers.contains_key(inp) {
                    if let Some(&dt) = type_map.get(inp) {
                        if dt != DType::Float {
                            return false;
                        }
                    }
                }
            }
            // Spatial ops (Conv, MaxPool, GlobalAvgPool) require 4D input.
            // If shapes are known at plan time, verify they are 4D.
            // If shapes are unknown (dynamic spatial dims), allow them through —
            // ensure_compiled will resolve shapes from runtime tensors and fall back
            // to CPU if they turn out incompatible.
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
                    // else: shape unknown at plan time — allow through for lazy compilation
                } else {
                    return false;
                }
            }
            // MaxPool with auto_pad or asymmetric padding
            if *op == OpType::MaxPool {
                let auto_pad = get_attr_string(node, "auto_pad").unwrap_or_default();
                if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                    return false;
                }
                let pads = get_attr_ints(node, "pads").unwrap_or_default();
                if pads.len() >= 4 && (pads[0] != pads[2] || pads[1] != pads[3]) {
                    return false;
                }
            }
            // Conv and Gemm require static weights
            if matches!(op, OpType::Conv | OpType::Gemm) {
                if inputs.len() < 2 || !initializers.contains_key(&inputs[1]) {
                    return false;
                }
            }
            // MatMul with static B uses fully_connected; dynamic B uses batch_matrix_multiply
            // Both are fine, but we need shapes for batch_matrix_multiply
            if *op == OpType::MatMul && inputs.len() >= 2 && !initializers.contains_key(&inputs[1])
            {
                let a = inputs.first().and_then(|n| shape_map.get(n));
                let b = inputs.get(1).and_then(|n| shape_map.get(n));
                if a.is_none() || b.is_none() {
                    return false;
                }
            }
            // Reshape/Flatten need known output shapes at compile time;
            // ensure_compiled will infer them from runtime shapes if not
            // available at plan time.
            // BatchNorm requires all params in initializers
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
            // Concat: only allow when all inputs and output are 4D
            // (non-4D concat axis semantics differ from spatial NCHW layout)
            if *op == OpType::Concat {
                for name in inputs.iter().chain(node.output.iter()) {
                    if name.is_empty() {
                        continue;
                    }
                    if let Some(s) = shape_map.get(name) {
                        if s.len() != 4 {
                            return false;
                        }
                    }
                    // else: shape unknown — allow through for lazy compilation
                }
            }
            true
        })
        .collect();

    // Find contiguous runs of compatible nodes (min length 2)
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

    // Build shape_map with Vec<usize> values for the subgraph builder
    let shape_map_vec: HashMap<String, Vec<usize>> = shape_map
        .iter()
        .map(|(k, v)| (k.clone(), v.to_vec()))
        .collect();

    // Replace runs with XnnpackSubgraph nodes (process in reverse to preserve indices)
    // Collect all node inputs/outputs for determining required outputs
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
        // Collect all outputs produced by this run
        let mut run_outputs: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for meta in &node_meta[run.clone()] {
            if let Some((_, _, node)) = meta {
                for out in &node.output {
                    if !out.is_empty() {
                        run_outputs.insert(out.as_str());
                    }
                }
            }
        }

        // Find which of those outputs are consumed by nodes OUTSIDE the run
        // or are plan outputs
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
        // Always include the last op's outputs (they may be plan-level outputs)
        if let Some(meta) = node_meta[run.end - 1].as_ref() {
            for out in &meta.2.output {
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
                    outputs: node.output.clone(),
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

        // Collect initializers needed by captured ops
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

        // Remove the nodes and store as fallback for lazy compilation failure
        let removed: Vec<_> = nodes.drain(run.clone()).collect();
        subgraph.fallback_nodes = removed;
        nodes.insert(run.start, PlanNode::XnnpackSubgraph(Box::new(subgraph)));
    }

    Ok(nodes)
}
