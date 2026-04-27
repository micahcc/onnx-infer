use std::collections::HashMap;

use crate::layers::OpType;
use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;

// ─── ElemType ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    Float,
    Uint8,
    Int8,
    Int32,
    Int64,
    String,
    Double,
    Bool,
    Unknown(i32),
}

impl ElemType {
    pub fn from_onnx(v: i32) -> Self {
        match v {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            11 => Self::Double,
            other => Self::Unknown(other),
        }
    }

    pub fn to_dtype(self) -> DType {
        match self {
            Self::Int32 | Self::Int64 => DType::Int64,
            Self::String => DType::String,
            _ => DType::Float,
        }
    }

    pub fn is_int(self) -> bool {
        matches!(self, Self::Int32 | Self::Int64)
    }
}

// ─── Attr / Attrs (parsing helpers — NOT stored on Node long-term) ─────────

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Attr {
    Int(i64),
    Float(f32),
    String(String),
    Tensor(Tensor),
    Graph(Box<Graph>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<Vec<u8>>),
}

#[derive(Debug, Clone)]
struct Attrs(HashMap<String, Attr>);

impl Attrs {
    pub fn get(&self, name: &str) -> Option<&Attr> {
        self.0.get(name)
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.0.get(name)? {
            Attr::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.0.get(name)? {
            Attr::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_ints(&self, name: &str) -> Option<Vec<i64>> {
        match self.0.get(name)? {
            Attr::Ints(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn get_string(&self, name: &str) -> Option<String> {
        match self.0.get(name)? {
            Attr::String(v) if !v.is_empty() => Some(v.clone()),
            _ => None,
        }
    }
}

// ─── NodeOp — strongly typed, opset-consistent representation ──────────────

/// Strongly typed op with all parameters resolved at parse time.
///
/// Old-opset attributes that moved to inputs in later opsets are converted to
/// initializer tensors by `normalize_for_opset` and wired as inputs. NodeOp
/// stores only "true" configuration attributes — never parameters that come
/// from tensor inputs after normalization.
#[derive(Debug, Clone)]
pub enum NodeOp {
    // Parameterless (inputs only, no configuration)
    Abs,
    Acos,
    Acosh,
    And,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Ceil,
    Clip,
    Cos,
    Cosh,
    Equal,
    Erf,
    Exp,
    Expand,
    Floor,
    GlobalAveragePool,
    Greater,
    Identity,
    IsInf,
    IsNaN,
    Less,
    Log,
    MatMul,
    Max,
    Min,
    Neg,
    NonMaxSuppression,
    NonZero,
    Not,
    PRelu,
    QLinearAdd,
    QLinearMatMul,
    QLinearGlobalAveragePool,
    QuantizeLinear,
    Range,
    Reciprocal,
    Relu,
    Reshape,
    Round,
    Shape,
    Sigmoid,
    Sign,
    Sin,
    Sinh,
    Slice,
    Softplus,
    Softsign,
    Sqrt,
    Squeeze,
    Sub,
    Sum,
    Tan,
    Tanh,
    Tile,
    Unsqueeze,
    Where,

    // Binary arithmetic (legacy_broadcast for opset < 7)
    Add {
        legacy_broadcast: bool,
        axis: i64,
    },
    Div {
        legacy_broadcast: bool,
        axis: i64,
    },
    Mul {
        legacy_broadcast: bool,
        axis: i64,
    },

    // Parameterized
    ArgMax {
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
    },
    AveragePool {
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        auto_pad: String,
        count_include_pad: i64,
    },
    BatchNormalization {
        epsilon: f32,
    },
    BatchNormalization2d {
        epsilon: f32,
    },
    Cast {
        to: i64,
    },
    CategoryMapper {
        cats_strings: Vec<Vec<u8>>,
        cats_int64s: Vec<i64>,
        default_int64: i64,
    },
    Celu {
        alpha: f32,
    },
    Compress {
        axis: Option<i64>,
    },
    Concat {
        axis: i64,
    },
    Constant {
        value: Tensor,
    },
    ConstantOfShape {
        value: Option<Tensor>,
    },
    Conv {
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        auto_pad: String,
    },
    ConvTranspose {
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
    },
    DequantizeLinear {
        axis: i64,
    },
    Dropout,
    Elu {
        alpha: f32,
    },
    Flatten {
        axis: i64,
    },
    Gather {
        axis: i64,
    },
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    Hardmax {
        axis: i64,
    },
    If {
        then_branch: Box<Graph>,
        else_branch: Box<Graph>,
    },
    LayoutTranspose {
        perm: Vec<i64>,
    },
    LeakyRelu {
        alpha: f32,
    },
    Loop {
        body: Box<Graph>,
    },
    Lrn {
        size: i64,
        alpha: f32,
        beta: f32,
        bias: f32,
    },
    Lstm {
        hidden_size: i64,
        direction: String,
    },
    MaxPool {
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        auto_pad: String,
    },
    QLinearConv {
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        auto_pad: String,
    },
    ReduceMax {
        keepdims: bool,
    },
    ReduceMean {
        keepdims: bool,
        noop_with_empty_axes: bool,
    },
    ReduceMin {
        keepdims: bool,
    },
    ReduceSum {
        keepdims: bool,
        noop_with_empty_axes: bool,
    },
    Resize {
        mode: String,
        coordinate_transformation_mode: String,
        nearest_mode: String,
    },
    RoiAlign {
        mode: String,
        output_height: i64,
        output_width: i64,
        sampling_ratio: i64,
        spatial_scale: f32,
    },
    Scan {
        body: Box<Graph>,
        num_scan_inputs: i64,
        scan_input_directions: Vec<i64>,
        scan_output_directions: Vec<i64>,
    },
    ScatterElements {
        axis: i64,
    },
    Selu {
        alpha: f32,
        gamma: f32,
    },
    Softmax {
        axis: i64,
        coerce_2d: bool,
    },
    Split {
        axis: i64,
    },
    ThresholdedRelu {
        alpha: f32,
    },
    TopK {
        axis: i64,
        largest: bool,
    },
    Transpose {
        perm: Option<Vec<i64>>,
    },
    Upsample {
        mode: String,
    },
}

impl NodeOp {
    /// Map back to OpType for pattern-matching in graph_opt / plan / xnnpack.
    pub fn op_type(&self) -> OpType {
        match self {
            Self::Abs => OpType::Abs,
            Self::Acos => OpType::Acos,
            Self::Acosh => OpType::Acosh,
            Self::Add { .. } => OpType::Add,
            Self::And => OpType::And,
            Self::ArgMax { .. } => OpType::ArgMax,
            Self::Asin => OpType::Asin,
            Self::Asinh => OpType::Asinh,
            Self::Atan => OpType::Atan,
            Self::Atanh => OpType::Atanh,
            Self::AveragePool { .. } => OpType::AveragePool,
            Self::BatchNormalization { .. } => OpType::BatchNormalization,
            Self::BatchNormalization2d { .. } => OpType::BatchNormalization2d,
            Self::Cast { .. } => OpType::Cast,
            Self::CategoryMapper { .. } => OpType::CategoryMapper,
            Self::Ceil => OpType::Ceil,
            Self::Celu { .. } => OpType::Celu,
            Self::Clip => OpType::Clip,
            Self::Compress { .. } => OpType::Compress,
            Self::Concat { .. } => OpType::Concat,
            Self::Constant { .. } => OpType::Constant,
            Self::ConstantOfShape { .. } => OpType::ConstantOfShape,
            Self::Conv { .. } => OpType::Conv,
            Self::ConvTranspose { .. } => OpType::ConvTranspose,
            Self::Cos => OpType::Cos,
            Self::Cosh => OpType::Cosh,
            Self::DequantizeLinear { .. } => OpType::DequantizeLinear,
            Self::Div { .. } => OpType::Div,
            Self::Dropout => OpType::Dropout,
            Self::Elu { .. } => OpType::Elu,
            Self::Equal => OpType::Equal,
            Self::Erf => OpType::Erf,
            Self::Exp => OpType::Exp,
            Self::Expand => OpType::Expand,
            Self::Flatten { .. } => OpType::Flatten,
            Self::Floor => OpType::Floor,
            Self::Gather { .. } => OpType::Gather,
            Self::Gemm { .. } => OpType::Gemm,
            Self::GlobalAveragePool => OpType::GlobalAveragePool,
            Self::Greater => OpType::Greater,
            Self::HardSigmoid { .. } => OpType::HardSigmoid,
            Self::Hardmax { .. } => OpType::Hardmax,
            Self::Identity => OpType::Identity,
            Self::If { .. } => OpType::If,
            Self::IsInf => OpType::IsInf,
            Self::IsNaN => OpType::IsNaN,
            Self::LayoutTranspose { .. } => OpType::LayoutTranspose,
            Self::LeakyRelu { .. } => OpType::LeakyRelu,
            Self::Less => OpType::Less,
            Self::Log => OpType::Log,
            Self::Loop { .. } => OpType::Loop,
            Self::Lrn { .. } => OpType::Lrn,
            Self::Lstm { .. } => OpType::Lstm,
            Self::MatMul => OpType::MatMul,
            Self::Max => OpType::Max,
            Self::MaxPool { .. } => OpType::MaxPool,
            Self::Min => OpType::Min,
            Self::Mul { .. } => OpType::Mul,
            Self::Neg => OpType::Neg,
            Self::NonMaxSuppression => OpType::NonMaxSuppression,
            Self::NonZero => OpType::NonZero,
            Self::Not => OpType::Not,
            Self::PRelu => OpType::PRelu,
            Self::QLinearAdd => OpType::QLinearAdd,
            Self::QLinearConv { .. } => OpType::QLinearConv,
            Self::QLinearGlobalAveragePool => OpType::QLinearGlobalAveragePool,
            Self::QLinearMatMul => OpType::QLinearMatMul,
            Self::QuantizeLinear => OpType::QuantizeLinear,
            Self::Range => OpType::Range,
            Self::Reciprocal => OpType::Reciprocal,
            Self::ReduceMax { .. } => OpType::ReduceMax,
            Self::ReduceMean { .. } => OpType::ReduceMean,
            Self::ReduceMin { .. } => OpType::ReduceMin,
            Self::ReduceSum { .. } => OpType::ReduceSum,
            Self::Relu => OpType::Relu,
            Self::Reshape => OpType::Reshape,
            Self::Resize { .. } => OpType::Resize,
            Self::RoiAlign { .. } => OpType::RoiAlign,
            Self::Round => OpType::Round,
            Self::Scan { .. } => OpType::Scan,
            Self::ScatterElements { .. } => OpType::ScatterElements,
            Self::Selu { .. } => OpType::Selu,
            Self::Shape => OpType::Shape,
            Self::Sigmoid => OpType::Sigmoid,
            Self::Sign => OpType::Sign,
            Self::Sin => OpType::Sin,
            Self::Sinh => OpType::Sinh,
            Self::Slice => OpType::Slice,
            Self::Softmax { .. } => OpType::Softmax,
            Self::Softplus => OpType::Softplus,
            Self::Softsign => OpType::Softsign,
            Self::Split { .. } => OpType::Split,
            Self::Sqrt => OpType::Sqrt,
            Self::Squeeze => OpType::Squeeze,
            Self::Sub => OpType::Sub,
            Self::Sum => OpType::Sum,
            Self::Tan => OpType::Tan,
            Self::Tanh => OpType::Tanh,
            Self::ThresholdedRelu { .. } => OpType::ThresholdedRelu,
            Self::Tile => OpType::Tile,
            Self::TopK { .. } => OpType::TopK,
            Self::Transpose { .. } => OpType::Transpose,
            Self::Unsqueeze => OpType::Unsqueeze,
            Self::Upsample { .. } => OpType::Upsample,
            Self::Where => OpType::Where,
        }
    }
}

// ─── Node ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Node {
    pub op: NodeOp,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl Node {
    /// Shorthand for `self.op.op_type()`.
    pub fn op_type(&self) -> OpType {
        self.op.op_type()
    }
}

impl NodeOp {
    /// Return perm for Transpose or LayoutTranspose, None otherwise.
    pub fn perm(&self) -> Option<&[i64]> {
        match self {
            Self::Transpose { perm } => perm.as_deref(),
            Self::LayoutTranspose { perm } => Some(perm),
            _ => None,
        }
    }

    /// Return subgraphs (body/branches) if this op contains them.
    pub fn subgraphs(&self) -> Vec<&Graph> {
        match self {
            Self::Loop { body } => vec![body],
            Self::If {
                then_branch,
                else_branch,
            } => vec![then_branch, else_branch],
            Self::Scan { body, .. } => vec![body],
            _ => vec![],
        }
    }
}

// ─── ValueInfo / Graph ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub elem_type: ElemType,
    pub shape: Option<Dims>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub initializers: HashMap<String, Tensor>,
    pub opset_version: i64,
}

// ─── Conversion ────────────────────────────────────────────────────────────

pub fn convert_graph(graph: &crate::onnx::GraphProto) -> Result<Graph> {
    convert_graph_with_opset(graph, 0)
}

pub fn convert_graph_with_opset(
    graph: &crate::onnx::GraphProto,
    opset_version: i64,
) -> Result<Graph> {
    let mut initializers = HashMap::new();
    for init in &graph.initializer {
        if !init.name.is_empty() {
            initializers.insert(init.name.clone(), Tensor::from_proto(init)?);
        }
    }

    let inputs = graph.input.iter().map(convert_value_info).collect();
    let outputs = graph.output.iter().map(convert_value_info).collect();

    // Parse → normalize (operates on raw attrs) → finalize (derives NodeOp)
    let mut raw_nodes: Vec<RawNode> = graph
        .node
        .iter()
        .map(|n| parse_raw_node(n, opset_version))
        .collect::<Result<_>>()?;

    normalize_for_opset(opset_version, &mut raw_nodes, &mut initializers);

    let nodes = raw_nodes
        .into_iter()
        .map(|raw| finalize_node(raw, opset_version))
        .collect();

    Ok(Graph {
        nodes,
        inputs,
        outputs,
        initializers,
        opset_version,
    })
}

fn convert_value_info(vi: &crate::onnx::ValueInfoProto) -> ValueInfo {
    let name = vi.name.clone();
    let (elem_type, shape) = vi
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .map(|v| match v {
            crate::onnx::type_proto::Value::TensorType(tt) => {
                let shape = tt.shape.as_ref().and_then(|s| {
                    if s.dim.is_empty() {
                        return None;
                    }
                    let mut dims = Dims::new();
                    for d in &s.dim {
                        match &d.value {
                            Some(crate::onnx::tensor_shape_proto::dimension::Value::DimValue(
                                v,
                            )) if *v > 0 => {
                                dims.push(*v as usize);
                            }
                            _ => dims.push(0),
                        }
                    }
                    Some(dims)
                });
                (ElemType::from_onnx(tt.elem_type), shape)
            }
            _ => (ElemType::Unknown(0), None),
        })
        .unwrap_or((ElemType::Unknown(0), None));
    ValueInfo {
        name,
        elem_type,
        shape,
    }
}

/// Extract tensor shape from an ONNX ValueInfoProto.
/// Dynamic or missing dimensions are returned as 1.
pub fn extract_tensor_shape(
    value_info: &crate::onnx::ValueInfoProto,
) -> anyhow::Result<Vec<usize>> {
    let type_proto = value_info
        .r#type
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("ValueInfo has no type"))?;
    let value = type_proto
        .value
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Type has no value"))?;

    let tensor_type = match value {
        crate::onnx::type_proto::Value::TensorType(t) => t,
        _ => anyhow::bail!("Type is not a tensor"),
    };

    let shape_proto = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tensor has no shape"))?;

    Ok(shape_proto
        .dim
        .iter()
        .map(|d| {
            use crate::onnx::tensor_shape_proto::dimension::Value as DimValue;
            match d.value.as_ref() {
                Some(DimValue::DimValue(v)) => *v as usize,
                _ => 1,
            }
        })
        .collect())
}

/// Temporary node representation used during parsing / normalization.
/// After `normalize_for_opset` runs, these are converted to final `Node` values.
struct RawNode {
    op_type: OpType,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Attrs,
}

fn parse_raw_node(node: &crate::onnx::NodeProto, opset: i64) -> Result<RawNode> {
    let op_type =
        OpType::parse(&node.op_type).map_err(|s| anyhow::anyhow!("unsupported operator: {s}"))?;
    let attrs_map: HashMap<String, Attr> = node
        .attribute
        .iter()
        .map(|a| convert_attr(a, opset))
        .collect::<Result<_>>()?;
    Ok(RawNode {
        op_type,
        name: node.name.clone(),
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        attrs: Attrs(attrs_map),
    })
}

fn finalize_node(raw: RawNode, opset: i64) -> Node {
    let op = derive_node_op(raw.op_type, &raw.attrs, opset);
    Node {
        op,
        name: raw.name,
        inputs: raw.inputs,
        outputs: raw.outputs,
    }
}

fn convert_attr(attr: &crate::onnx::AttributeProto, opset: i64) -> Result<(String, Attr)> {
    let name = attr.name.clone();
    let val = match attr.r#type {
        1 => Attr::Float(attr.f),
        2 => Attr::Int(attr.i),
        3 => Attr::String(String::from_utf8_lossy(&attr.s).to_string()),
        4 => {
            let t = attr.t.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Attribute '{name}' has tensor type but no tensor")
            })?;
            Attr::Tensor(Tensor::from_proto(t)?)
        }
        5 => {
            let g = attr
                .g
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Attribute '{name}' has graph type but no graph"))?;
            Attr::Graph(Box::new(convert_graph_with_opset(g, opset)?))
        }
        6 => Attr::Floats(attr.floats.clone()),
        7 => Attr::Ints(attr.ints.clone()),
        8 => Attr::Strings(attr.strings.clone()),
        _ => {
            // type 0 (UNDEFINED) or unknown: detect from populated fields
            if let Some(t) = &attr.t {
                Attr::Tensor(Tensor::from_proto(t)?)
            } else if let Some(g) = &attr.g {
                Attr::Graph(Box::new(convert_graph_with_opset(g, opset)?))
            } else if !attr.ints.is_empty() {
                Attr::Ints(attr.ints.clone())
            } else if !attr.floats.is_empty() {
                Attr::Floats(attr.floats.clone())
            } else if !attr.strings.is_empty() {
                Attr::Strings(attr.strings.clone())
            } else if !attr.s.is_empty() {
                Attr::String(String::from_utf8_lossy(&attr.s).to_string())
            } else if attr.f != 0.0 {
                Attr::Float(attr.f)
            } else {
                Attr::Int(attr.i)
            }
        }
    };
    Ok((name, val))
}

// ─── NodeOp derivation ────────────────────────────────────────────────────

/// Derive a strongly-typed NodeOp from the parsed OpType + Attrs + opset.
/// Opset-dependent defaults are resolved here so downstream code never needs
/// to know the opset version.
fn derive_node_op(op_type: OpType, a: &Attrs, opset: i64) -> NodeOp {
    match op_type {
        // Simple parameterless
        OpType::Abs => NodeOp::Abs,
        OpType::Acos => NodeOp::Acos,
        OpType::Acosh => NodeOp::Acosh,
        OpType::And => NodeOp::And,
        OpType::Asin => NodeOp::Asin,
        OpType::Asinh => NodeOp::Asinh,
        OpType::Atan => NodeOp::Atan,
        OpType::Atanh => NodeOp::Atanh,
        OpType::Ceil => NodeOp::Ceil,
        OpType::Clip => NodeOp::Clip,
        OpType::Cos => NodeOp::Cos,
        OpType::Cosh => NodeOp::Cosh,
        OpType::Equal => NodeOp::Equal,
        OpType::Erf => NodeOp::Erf,
        OpType::Exp => NodeOp::Exp,
        OpType::Expand => NodeOp::Expand,
        OpType::Floor => NodeOp::Floor,
        OpType::GlobalAveragePool => NodeOp::GlobalAveragePool,
        OpType::Greater => NodeOp::Greater,
        OpType::Identity => NodeOp::Identity,
        OpType::IsInf => NodeOp::IsInf,
        OpType::IsNaN => NodeOp::IsNaN,
        OpType::Less => NodeOp::Less,
        OpType::Log => NodeOp::Log,
        OpType::MatMul => NodeOp::MatMul,
        OpType::Max => NodeOp::Max,
        OpType::Min => NodeOp::Min,
        OpType::Neg => NodeOp::Neg,
        OpType::NonMaxSuppression => NodeOp::NonMaxSuppression,
        OpType::NonZero => NodeOp::NonZero,
        OpType::Not => NodeOp::Not,
        OpType::PRelu => NodeOp::PRelu,
        OpType::QLinearAdd => NodeOp::QLinearAdd,
        OpType::QLinearMatMul => NodeOp::QLinearMatMul,
        OpType::QLinearGlobalAveragePool => NodeOp::QLinearGlobalAveragePool,
        OpType::QuantizeLinear => NodeOp::QuantizeLinear,
        OpType::Range => NodeOp::Range,
        OpType::Reciprocal => NodeOp::Reciprocal,
        OpType::Relu => NodeOp::Relu,
        OpType::Reshape => NodeOp::Reshape,
        OpType::Round => NodeOp::Round,
        OpType::Shape => NodeOp::Shape,
        OpType::Sigmoid => NodeOp::Sigmoid,
        OpType::Sign => NodeOp::Sign,
        OpType::Sin => NodeOp::Sin,
        OpType::Sinh => NodeOp::Sinh,
        OpType::Slice => NodeOp::Slice,
        OpType::Softplus => NodeOp::Softplus,
        OpType::Softsign => NodeOp::Softsign,
        OpType::Sqrt => NodeOp::Sqrt,
        OpType::Squeeze => NodeOp::Squeeze,
        OpType::Sub => NodeOp::Sub,
        OpType::Sum => NodeOp::Sum,
        OpType::Tan => NodeOp::Tan,
        OpType::Tanh => NodeOp::Tanh,
        OpType::Tile => NodeOp::Tile,
        OpType::Unsqueeze => NodeOp::Unsqueeze,
        OpType::Where => NodeOp::Where,

        // Binary arithmetic with legacy broadcast
        OpType::Add => NodeOp::Add {
            legacy_broadcast: a.get_int("broadcast").unwrap_or(0) != 0,
            axis: a.get_int("axis").unwrap_or(0),
        },
        OpType::Div => NodeOp::Div {
            legacy_broadcast: a.get_int("broadcast").unwrap_or(0) != 0,
            axis: a.get_int("axis").unwrap_or(0),
        },
        OpType::Mul => NodeOp::Mul {
            legacy_broadcast: a.get_int("broadcast").unwrap_or(0) != 0,
            axis: a.get_int("axis").unwrap_or(0),
        },

        // Parameterized ops
        OpType::ArgMax => NodeOp::ArgMax {
            axis: a.get_int("axis").unwrap_or(0),
            keepdims: a.get_int("keepdims").unwrap_or(1) != 0,
            select_last_index: a.get_int("select_last_index").unwrap_or(0) != 0,
        },
        OpType::AveragePool => NodeOp::AveragePool {
            kernel_shape: a.get_ints("kernel_shape").unwrap_or_default(),
            strides: a.get_ints("strides").unwrap_or_else(|| vec![1, 1]),
            pads: a.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]),
            auto_pad: a.get_string("auto_pad").unwrap_or_default(),
            count_include_pad: a.get_int("count_include_pad").unwrap_or(0),
        },
        OpType::BatchNormalization => NodeOp::BatchNormalization {
            epsilon: a.get_float("epsilon").unwrap_or(1e-5),
        },
        OpType::BatchNormalization2d => NodeOp::BatchNormalization2d {
            epsilon: a.get_float("epsilon").unwrap_or(1e-5),
        },
        OpType::Cast => NodeOp::Cast {
            to: a.get_int("to").unwrap_or(1),
        },
        OpType::CategoryMapper => NodeOp::CategoryMapper {
            cats_strings: match a.get("cats_strings") {
                Some(Attr::Strings(v)) => v.clone(),
                _ => vec![],
            },
            cats_int64s: match a.get("cats_int64s") {
                Some(Attr::Ints(v)) => v.clone(),
                _ => vec![],
            },
            default_int64: a.get_int("default_int64").unwrap_or(-1),
        },
        OpType::Celu => NodeOp::Celu {
            alpha: a.get_float("alpha").unwrap_or(1.0),
        },
        OpType::Compress => NodeOp::Compress {
            axis: a.get_int("axis"),
        },
        OpType::Concat => NodeOp::Concat {
            axis: a.get_int("axis").unwrap_or(0),
        },
        OpType::Constant => NodeOp::Constant {
            value: match a.get("value") {
                Some(Attr::Tensor(t)) => t.clone(),
                _ => Tensor::default(),
            },
        },
        OpType::ConstantOfShape => NodeOp::ConstantOfShape {
            value: match a.get("value") {
                Some(Attr::Tensor(t)) => Some(t.clone()),
                _ => None,
            },
        },
        OpType::Conv => NodeOp::Conv {
            kernel_shape: a.get_ints("kernel_shape").unwrap_or_default(),
            strides: a.get_ints("strides").unwrap_or_else(|| vec![1, 1]),
            pads: a.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: a.get_ints("dilations").unwrap_or_else(|| vec![1, 1]),
            group: a.get_int("group").unwrap_or(1),
            auto_pad: a.get_string("auto_pad").unwrap_or_default(),
        },
        OpType::ConvTranspose => NodeOp::ConvTranspose {
            strides: a.get_ints("strides").unwrap_or_else(|| vec![1, 1]),
            pads: a.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: a.get_ints("dilations").unwrap_or_else(|| vec![1, 1]),
            group: a.get_int("group").unwrap_or(1),
        },
        OpType::DequantizeLinear => NodeOp::DequantizeLinear {
            axis: a.get_int("axis").unwrap_or(1),
        },
        OpType::Dropout => NodeOp::Dropout,
        OpType::Elu => NodeOp::Elu {
            alpha: a.get_float("alpha").unwrap_or(1.0),
        },
        OpType::Flatten => NodeOp::Flatten {
            axis: a.get_int("axis").unwrap_or(1),
        },
        OpType::Gather => NodeOp::Gather {
            axis: a.get_int("axis").unwrap_or(0),
        },
        OpType::Gemm => NodeOp::Gemm {
            alpha: a.get_float("alpha").unwrap_or(1.0),
            beta: a.get_float("beta").unwrap_or(1.0),
            trans_a: a.get_int("transA").unwrap_or(0) != 0,
            trans_b: a.get_int("transB").unwrap_or(0) != 0,
        },
        OpType::HardSigmoid => NodeOp::HardSigmoid {
            alpha: a.get_float("alpha").unwrap_or(0.2),
            beta: a.get_float("beta").unwrap_or(0.5),
        },
        OpType::Hardmax => NodeOp::Hardmax {
            axis: a.get_int("axis").unwrap_or(-1),
        },
        OpType::If => {
            let then_branch = match a.get("then_branch") {
                Some(Attr::Graph(g)) => (**g).clone(),
                _ => Graph {
                    nodes: vec![],
                    inputs: vec![],
                    outputs: vec![],
                    initializers: HashMap::new(),
                    opset_version: opset,
                },
            };
            let else_branch = match a.get("else_branch") {
                Some(Attr::Graph(g)) => (**g).clone(),
                _ => Graph {
                    nodes: vec![],
                    inputs: vec![],
                    outputs: vec![],
                    initializers: HashMap::new(),
                    opset_version: opset,
                },
            };
            NodeOp::If {
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            }
        }
        OpType::LayoutTranspose => NodeOp::LayoutTranspose {
            perm: a.get_ints("perm").unwrap_or_default(),
        },
        OpType::LeakyRelu => NodeOp::LeakyRelu {
            alpha: a.get_float("alpha").unwrap_or(0.01),
        },
        OpType::Loop => {
            let body = match a.get("body") {
                Some(Attr::Graph(g)) => (**g).clone(),
                _ => Graph {
                    nodes: vec![],
                    inputs: vec![],
                    outputs: vec![],
                    initializers: HashMap::new(),
                    opset_version: opset,
                },
            };
            NodeOp::Loop {
                body: Box::new(body),
            }
        }
        OpType::Lrn => NodeOp::Lrn {
            size: a.get_int("size").unwrap_or(1),
            alpha: a.get_float("alpha").unwrap_or(0.0001),
            beta: a.get_float("beta").unwrap_or(0.75),
            bias: a.get_float("bias").unwrap_or(1.0),
        },
        OpType::Lstm => NodeOp::Lstm {
            hidden_size: a.get_int("hidden_size").unwrap_or(1),
            direction: a.get_string("direction").unwrap_or_default(),
        },
        OpType::MaxPool => NodeOp::MaxPool {
            kernel_shape: a.get_ints("kernel_shape").unwrap_or_default(),
            strides: a.get_ints("strides").unwrap_or_else(|| vec![1, 1]),
            pads: a.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]),
            auto_pad: a.get_string("auto_pad").unwrap_or_default(),
        },
        OpType::QLinearConv => NodeOp::QLinearConv {
            kernel_shape: a.get_ints("kernel_shape").unwrap_or_default(),
            strides: a.get_ints("strides").unwrap_or_else(|| vec![1, 1]),
            pads: a.get_ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]),
            dilations: a.get_ints("dilations").unwrap_or_else(|| vec![1, 1]),
            group: a.get_int("group").unwrap_or(1),
            auto_pad: a.get_string("auto_pad").unwrap_or_default(),
        },
        OpType::ReduceMax => NodeOp::ReduceMax {
            keepdims: a.get_int("keepdims").unwrap_or(1) != 0,
        },
        OpType::ReduceMean => NodeOp::ReduceMean {
            keepdims: a.get_int("keepdims").unwrap_or(1) != 0,
            noop_with_empty_axes: a.get_int("noop_with_empty_axes").unwrap_or(0) != 0,
        },
        OpType::ReduceMin => NodeOp::ReduceMin {
            keepdims: a.get_int("keepdims").unwrap_or(1) != 0,
        },
        OpType::ReduceSum => NodeOp::ReduceSum {
            keepdims: a.get_int("keepdims").unwrap_or(1) != 0,
            noop_with_empty_axes: a.get_int("noop_with_empty_axes").unwrap_or(0) != 0,
        },
        OpType::Resize => NodeOp::Resize {
            mode: a
                .get_string("mode")
                .unwrap_or_else(|| "nearest".to_string()),
            coordinate_transformation_mode: a
                .get_string("coordinate_transformation_mode")
                .unwrap_or_default(),
            nearest_mode: a.get_string("nearest_mode").unwrap_or_default(),
        },
        OpType::RoiAlign => NodeOp::RoiAlign {
            mode: a.get_string("mode").unwrap_or_else(|| "avg".to_string()),
            output_height: a.get_int("output_height").unwrap_or(1),
            output_width: a.get_int("output_width").unwrap_or(1),
            sampling_ratio: a.get_int("sampling_ratio").unwrap_or(0),
            spatial_scale: a.get_float("spatial_scale").unwrap_or(1.0),
        },
        OpType::Scan => {
            let body = match a.get("body") {
                Some(Attr::Graph(g)) => (**g).clone(),
                _ => Graph {
                    nodes: vec![],
                    inputs: vec![],
                    outputs: vec![],
                    initializers: HashMap::new(),
                    opset_version: opset,
                },
            };
            NodeOp::Scan {
                body: Box::new(body),
                num_scan_inputs: a.get_int("num_scan_inputs").unwrap_or(0),
                scan_input_directions: a.get_ints("scan_input_directions").unwrap_or_default(),
                scan_output_directions: a.get_ints("scan_output_directions").unwrap_or_default(),
            }
        }
        OpType::ScatterElements => NodeOp::ScatterElements {
            axis: a.get_int("axis").unwrap_or(0),
        },
        OpType::Selu => NodeOp::Selu {
            alpha: a.get_float("alpha").unwrap_or(1.673_263_2),
            gamma: a.get_float("gamma").unwrap_or(1.050_701),
        },
        OpType::Softmax => {
            let default_axis = if opset > 0 && opset < 13 { 1 } else { -1 };
            let coerce_2d = opset > 0 && opset < 13;
            NodeOp::Softmax {
                axis: a.get_int("axis").unwrap_or(default_axis),
                coerce_2d,
            }
        }
        OpType::Split => NodeOp::Split {
            axis: a.get_int("axis").unwrap_or(0),
        },
        OpType::ThresholdedRelu => NodeOp::ThresholdedRelu {
            alpha: a.get_float("alpha").unwrap_or(1.0),
        },
        OpType::TopK => NodeOp::TopK {
            axis: a.get_int("axis").unwrap_or(-1),
            largest: a.get_int("largest").unwrap_or(1) != 0,
        },
        OpType::Transpose => NodeOp::Transpose {
            perm: a.get_ints("perm"),
        },
        OpType::Upsample => NodeOp::Upsample {
            mode: a
                .get_string("mode")
                .unwrap_or_else(|| "nearest".to_string()),
        },
    }
}

// ─── Opset normalization ──────────────────────────────────────────────────

/// Convert old-opset attribute-based parameters into initializer tensor inputs
/// so that downstream layer code only needs to handle the input-based style.
/// Operates on raw parsed nodes before NodeOp derivation.
fn normalize_for_opset(
    opset: i64,
    nodes: &mut [RawNode],
    initializers: &mut HashMap<String, Tensor>,
) {
    if opset == 0 {
        return;
    }
    let mut counter = 0usize;
    let mut new_initializers: Vec<(String, Tensor)> = Vec::new();

    let mut gen_name = |prefix: &str| -> String {
        counter += 1;
        format!("__opset_norm_{prefix}_{counter}__")
    };

    for node in nodes.iter_mut() {
        match node.op_type {
            // Dropout → Identity
            OpType::Dropout => {
                node.op_type = OpType::Identity;
                node.inputs.truncate(1);
                node.attrs.0.clear();
            }

            // Squeeze: opset <13 has axes as attribute
            OpType::Squeeze if opset < 13 => {
                if let Some(Attr::Ints(axes_vec)) = node.attrs.0.remove("axes") {
                    if !axes_vec.is_empty() {
                        let name = gen_name("squeeze_axes");
                        let len = axes_vec.len();
                        new_initializers
                            .push((name.clone(), Tensor::new_i64(crate::dims![len], axes_vec)));
                        while node.inputs.len() < 2 {
                            node.inputs.push(String::new());
                        }
                        node.inputs[1] = name;
                    }
                }
            }

            // Unsqueeze: opset <13 has axes as attribute
            OpType::Unsqueeze if opset < 13 => {
                if let Some(Attr::Ints(axes_vec)) = node.attrs.0.remove("axes") {
                    if !axes_vec.is_empty() {
                        let name = gen_name("unsqueeze_axes");
                        let len = axes_vec.len();
                        new_initializers
                            .push((name.clone(), Tensor::new_i64(crate::dims![len], axes_vec)));
                        while node.inputs.len() < 2 {
                            node.inputs.push(String::new());
                        }
                        node.inputs[1] = name;
                    }
                }
            }

            // Split: opset <13 has split sizes as attribute
            OpType::Split if opset < 13 => {
                if let Some(Attr::Ints(split_vec)) = node.attrs.0.remove("split") {
                    if !split_vec.is_empty() {
                        let name = gen_name("split_sizes");
                        let len = split_vec.len();
                        new_initializers
                            .push((name.clone(), Tensor::new_i64(crate::dims![len], split_vec)));
                        while node.inputs.len() < 2 {
                            node.inputs.push(String::new());
                        }
                        node.inputs[1] = name;
                    }
                }
            }

            // Slice: opset <10 has starts/ends/axes as attributes
            OpType::Slice if opset < 10 => {
                let starts = node.attrs.0.remove("starts");
                let ends = node.attrs.0.remove("ends");
                let axes = node.attrs.0.remove("axes");

                if let (Some(Attr::Ints(starts_vec)), Some(Attr::Ints(ends_vec))) = (starts, ends) {
                    let starts_name = gen_name("slice_starts");
                    let ends_name = gen_name("slice_ends");
                    new_initializers.push((
                        starts_name.clone(),
                        Tensor::new_i64(crate::dims![starts_vec.len()], starts_vec),
                    ));
                    new_initializers.push((
                        ends_name.clone(),
                        Tensor::new_i64(crate::dims![ends_vec.len()], ends_vec),
                    ));
                    node.inputs.truncate(1);
                    node.inputs.push(starts_name);
                    node.inputs.push(ends_name);

                    if let Some(Attr::Ints(axes_vec)) = axes {
                        let axes_name = gen_name("slice_axes");
                        new_initializers.push((
                            axes_name.clone(),
                            Tensor::new_i64(crate::dims![axes_vec.len()], axes_vec),
                        ));
                        node.inputs.push(axes_name);
                    }
                }
            }

            // Reshape: opset <5 has shape as attribute
            OpType::Reshape if opset < 5 => {
                if let Some(Attr::Ints(shape_vec)) = node.attrs.0.remove("shape") {
                    let name = gen_name("reshape_shape");
                    let len = shape_vec.len();
                    new_initializers
                        .push((name.clone(), Tensor::new_i64(crate::dims![len], shape_vec)));
                    while node.inputs.len() < 2 {
                        node.inputs.push(String::new());
                    }
                    node.inputs[1] = name;
                }
            }

            // Clip: opset <11 has min/max as attributes
            OpType::Clip if opset < 11 => {
                let min_val = match node.attrs.0.remove("min") {
                    Some(Attr::Float(v)) => v,
                    _ => f32::NEG_INFINITY,
                };
                let max_val = match node.attrs.0.remove("max") {
                    Some(Attr::Float(v)) => v,
                    _ => f32::INFINITY,
                };
                let min_name = gen_name("clip_min");
                let max_name = gen_name("clip_max");
                new_initializers.push((
                    min_name.clone(),
                    Tensor::new(crate::dims![1], vec![min_val]),
                ));
                new_initializers.push((
                    max_name.clone(),
                    Tensor::new(crate::dims![1], vec![max_val]),
                ));
                while node.inputs.len() < 3 {
                    node.inputs.push(String::new());
                }
                node.inputs[1] = min_name;
                node.inputs[2] = max_name;
            }

            // TopK: opset <10 has k as attribute
            OpType::TopK if opset < 10 => {
                if let Some(Attr::Int(k)) = node.attrs.0.remove("k") {
                    let name = gen_name("topk_k");
                    new_initializers
                        .push((name.clone(), Tensor::new_i64(crate::dims![1], vec![k])));
                    while node.inputs.len() < 2 {
                        node.inputs.push(String::new());
                    }
                    node.inputs[1] = name;
                }
            }

            // ReduceSum/Mean/Max/Min: opset <18 has axes as attribute
            OpType::ReduceSum | OpType::ReduceMean | OpType::ReduceMax | OpType::ReduceMin
                if opset < 18 =>
            {
                if let Some(Attr::Ints(axes_vec)) = node.attrs.0.remove("axes") {
                    if !axes_vec.is_empty() {
                        let name = gen_name("reduce_axes");
                        let len = axes_vec.len();
                        new_initializers
                            .push((name.clone(), Tensor::new_i64(crate::dims![len], axes_vec)));
                        while node.inputs.len() < 2 {
                            node.inputs.push(String::new());
                        }
                        node.inputs[1] = name;
                    }
                }
            }

            _ => {}
        }
    }

    for (name, tensor) in new_initializers {
        initializers.insert(name, tensor);
    }
}
