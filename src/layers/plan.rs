use std::collections::HashMap;

use crate::DType;
use crate::InferenceError;
use crate::ONNX_INT32;
use crate::ONNX_INT64;
use crate::Result;
use crate::Tensor;
use crate::get_attr_float;
use crate::get_attr_int;
use crate::get_attr_ints;
use crate::get_attr_string;
use crate::layers::Layer;
use crate::layers::OpType;
use crate::layers::add;
use crate::layers::auto_cast;
use crate::layers::batch_norm;
use crate::layers::cast;
use crate::layers::ceil;
use crate::layers::clip;
use crate::layers::concat;
use crate::layers::constant;
use crate::layers::constant_of_shape;
use crate::layers::conv;
use crate::layers::dequantize_linear;
use crate::layers::div;
use crate::layers::equal;
use crate::layers::exp;
use crate::layers::expand;
use crate::layers::flatten;
use crate::layers::floor;
use crate::layers::gather;
use crate::layers::gemm;
use crate::layers::global_avg_pool;
use crate::layers::greater;
use crate::layers::identity;
use crate::layers::if_op;
use crate::layers::leaky_relu;
use crate::layers::less;
use crate::layers::log;
use crate::layers::loop_op;
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
use crate::layers::reduce_min;
use crate::layers::relu;
use crate::layers::reshape;
use crate::layers::resize;
use crate::layers::roi_align;
use crate::layers::round;
use crate::layers::scatter_elements;
use crate::layers::shape_op;
use crate::layers::sigmoid;
use crate::layers::slice;
use crate::layers::softmax;
use crate::layers::split;
use crate::layers::sqrt;
use crate::layers::squeeze;
use crate::layers::sub;
use crate::layers::tanh;
use crate::layers::tile;
use crate::layers::topk;
use crate::layers::transpose;
use crate::layers::unsqueeze;
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
}

pub struct Plan {
    pub nodes: Vec<PlanNode>,
    pub initializers: HashMap<String, Tensor>,
    pub output_names: Vec<String>,
    pub shape_map: HashMap<String, Vec<usize>>,
    pub type_map: HashMap<String, DType>,
    pub tensor_pool: HashMap<String, Tensor>,
}

impl Plan {
    pub fn build(
        graph: &crate::onnx::GraphProto,
        input_sizes: &HashMap<String, Vec<usize>>,
    ) -> Result<Self> {
        Self::build_with_types(graph, input_sizes, &HashMap::new())
    }

    pub fn build_with_types(
        graph: &crate::onnx::GraphProto,
        input_sizes: &HashMap<String, Vec<usize>>,
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

        let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (name, tensor) in &initializers {
            shape_map.insert(name.clone(), tensor.dims.clone());
        }
        for (name, dims) in input_sizes {
            shape_map.insert(name.clone(), dims.clone());
        }

        let mut known_values: HashMap<String, Tensor> = HashMap::new();

        let mut nodes = Vec::new();
        let mut cast_counter = 0usize;

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
                if let Some(Some(expected_dt)) = expected.get(i)
                    && let Some(&actual_dt) = type_map.get(input_name)
                    && actual_dt != *expected_dt
                {
                    let cast_name = format!("__auto_cast_{cast_counter}__");
                    cast_counter += 1;
                    nodes.push(PlanNode::Single {
                        output: cast_name.clone(),
                        layer: Box::new(auto_cast::AutoCastF32::new(input_name.clone())),
                    });
                    type_map.insert(cast_name.clone(), DType::Float);
                    modified_inputs[i] = cast_name;
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
                && let Some(out_name) = out_name
            {
                shape_map.insert(out_name.clone(), shape);
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

            nodes.push(build_node(op, node, modified_inputs)?);
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
    shape_map: &HashMap<String, Vec<usize>>,
) -> Option<Tensor> {
    if op == OpType::Shape {
        let name = input_names.first().filter(|s| !s.is_empty())?;
        let shape = shape_map.get(name)?;
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Some(Tensor::new_i64(vec![dims.len()], dims));
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

    let plan_node = build_node(op, node, input_names.to_vec()).ok()?;
    if let PlanNode::Single { mut layer, .. } = plan_node {
        let mut output = Tensor::default();
        layer.execute(&temp_values, &mut output).ok()?;
        Some(output)
    } else {
        None
    }
}

pub fn build_node(op: OpType, node: &NodeProto, inputs: Vec<String>) -> Result<PlanNode> {
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

    let output = if node.output.is_empty() || node.output[0].is_empty() {
        String::new()
    } else {
        node.output[0].clone()
    };

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
        )),
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
            Box::new(conv::Conv::new(inputs, ks, st, pa, di, gr, ap))
        }
        OpType::MatMul => Box::new(matmul::MatMul::new(inputs)),
        OpType::Gemm => Box::new(gemm::Gemm::new(
            inputs,
            get_attr_float(node, "alpha").unwrap_or(1.0),
            get_attr_float(node, "beta").unwrap_or(1.0),
            get_attr_int(node, "transA").unwrap_or(0) != 0,
            get_attr_int(node, "transB").unwrap_or(0) != 0,
        )),
        OpType::MaxPool => {
            let ks = get_attr_ints(node, "kernel_shape").unwrap_or_default();
            let st = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pa = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let ap = get_attr_string(node, "auto_pad").unwrap_or_default();
            Box::new(maxpool::MaxPool::new(inputs, ks, st, pa, ap)?)
        }
        OpType::GlobalAveragePool => Box::new(global_avg_pool::GlobalAvgPool::new(inputs)),
        OpType::Flatten => Box::new(flatten::Flatten::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(1) as usize,
        )),
        OpType::Shape => Box::new(shape_op::Shape::new(inputs)),
        OpType::Gather => Box::new(gather::Gather::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(0),
        )),
        OpType::Unsqueeze => Box::new(unsqueeze::Unsqueeze::new(inputs, node)),
        OpType::Concat => Box::new(concat::Concat::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(0),
        )),
        OpType::Identity => Box::new(identity::Identity::new(inputs)),
        OpType::Cast => Box::new(cast::Cast::new(
            inputs,
            get_attr_int(node, "to").unwrap_or(1),
        )),
        OpType::Transpose => {
            let perm = get_attr_ints(node, "perm").map(|p| p.iter().map(|&v| v as usize).collect());
            Box::new(transpose::Transpose::new(inputs, perm))
        }
        OpType::Squeeze => Box::new(squeeze::Squeeze::new(inputs, node)),
        OpType::Slice => Box::new(slice::Slice::new(inputs)),
        OpType::Tile => Box::new(tile::Tile::new(inputs)),
        OpType::Resize => {
            let ct = get_attr_string(node, "coordinate_transformation_mode").unwrap_or_default();
            let nm = get_attr_string(node, "nearest_mode").unwrap_or_default();
            Box::new(resize::Resize::new(inputs, &ct, &nm))
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
        )),
        OpType::NonMaxSuppression => Box::new(nms::Nms::new(inputs)),
        OpType::QuantizeLinear => Box::new(quantize_linear::QuantizeLinear::new(inputs)),
        OpType::DequantizeLinear => Box::new(dequantize_linear::DequantizeLinear::new(
            inputs,
            get_attr_int(node, "axis").unwrap_or(1),
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
            let inner = conv::Conv::new(conv_inputs, ks, st, pa, di, gr, ap);
            Box::new(qlinear_conv::QLinearConv::new(inputs, inner))
        }
        OpType::QLinearAdd => Box::new(qlinear_add::QLinearAdd::new(inputs)),
        OpType::QLinearMatMul => {
            let inner = matmul::MatMul::new(vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()]);
            Box::new(qlinear_matmul::QLinearMatMul::new(inputs, inner))
        }
        OpType::QLinearGlobalAveragePool => {
            let inner = global_avg_pool::GlobalAvgPool::new(vec!["__qgap_x__".to_string()]);
            Box::new(qlinear_global_avg_pool::QLinearGlobalAvgPool::new(
                inputs, inner,
            ))
        }
        OpType::Loop | OpType::Split | OpType::If | OpType::TopK => unreachable!(),
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

    if node.output.is_empty() || node.output[0].is_empty() {
        return Ok(());
    }

    let expected = op.expected_input_dtypes();
    let mut to_cast: Vec<(usize, String)> = Vec::new();
    for (i, input_name) in node.input.iter().enumerate() {
        if input_name.is_empty() {
            continue;
        }
        if let Some(Some(expected_dt)) = expected.get(i)
            && let Some(tensor) = values.get(input_name)
            && tensor.dtype() != *expected_dt
        {
            to_cast.push((i, input_name.clone()));
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

    let mut plan_node = build_node(op, node, modified_inputs)?;
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
    }
}
