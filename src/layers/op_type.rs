use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::Tensor;
use crate::broadcast_shape;
use crate::dims;
use crate::onnx_ir::Attr;
use crate::onnx_ir::Node;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Abs,
    Acos,
    Acosh,
    Add,
    And,
    ArgMax,
    AveragePool,
    Asin,
    Asinh,
    Atan,
    Atanh,
    BatchNormalization,
    Cast,
    CategoryMapper,
    Ceil,
    Celu,
    Clip,
    Compress,
    Concat,
    Constant,
    ConstantOfShape,
    Conv,
    ConvTranspose,
    Cos,
    Cosh,
    DequantizeLinear,
    Div,
    Dropout,
    Elu,
    Equal,
    Erf,
    Expand,
    Exp,
    Flatten,
    Floor,
    Gather,
    Gemm,
    GlobalAveragePool,
    Greater,
    HardSigmoid,
    Hardmax,
    Identity,
    If,
    IsInf,
    IsNaN,
    Lstm,
    LeakyRelu,
    LayoutTranspose,
    Less,
    Log,
    Loop,
    Lrn,
    MatMul,
    Max,
    MaxPool,
    Min,
    Mul,
    Neg,
    NonMaxSuppression,
    NonZero,
    Not,
    PRelu,
    QLinearAdd,
    QLinearConv,
    QLinearGlobalAveragePool,
    QLinearMatMul,
    QuantizeLinear,
    Range,
    Reciprocal,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceSum,
    Relu,
    Reshape,
    Resize,
    RoiAlign,
    Round,
    Scan,
    ScatterElements,
    Selu,
    Shape,
    Sigmoid,
    Sign,
    Sin,
    Sinh,
    Slice,
    Softmax,
    Softplus,
    Softsign,
    Split,
    Sqrt,
    Squeeze,
    Sub,
    Sum,
    Tan,
    Tanh,
    ThresholdedRelu,
    Tile,
    TopK,
    Transpose,
    Unsqueeze,
    Upsample,
    Where,
}

impl OpType {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "Abs" => Ok(Self::Abs),
            "Acos" => Ok(Self::Acos),
            "Acosh" => Ok(Self::Acosh),
            "Add" => Ok(Self::Add),
            "And" => Ok(Self::And),
            "ArgMax" => Ok(Self::ArgMax),
            "AveragePool" => Ok(Self::AveragePool),
            "Asin" => Ok(Self::Asin),
            "Asinh" => Ok(Self::Asinh),
            "Atan" => Ok(Self::Atan),
            "Atanh" => Ok(Self::Atanh),
            "BatchNormalization" => Ok(Self::BatchNormalization),
            "Cast" => Ok(Self::Cast),
            "CategoryMapper" => Ok(Self::CategoryMapper),
            "Ceil" => Ok(Self::Ceil),
            "Celu" => Ok(Self::Celu),
            "Clip" => Ok(Self::Clip),
            "Compress" => Ok(Self::Compress),
            "Concat" => Ok(Self::Concat),
            "Constant" => Ok(Self::Constant),
            "ConstantOfShape" => Ok(Self::ConstantOfShape),
            "Conv" => Ok(Self::Conv),
            "ConvTranspose" => Ok(Self::ConvTranspose),
            "Cos" => Ok(Self::Cos),
            "Cosh" => Ok(Self::Cosh),
            "DequantizeLinear" => Ok(Self::DequantizeLinear),
            "Div" => Ok(Self::Div),
            "Dropout" => Ok(Self::Dropout),
            "Elu" => Ok(Self::Elu),
            "Equal" => Ok(Self::Equal),
            "Erf" => Ok(Self::Erf),
            "Expand" => Ok(Self::Expand),
            "Exp" => Ok(Self::Exp),
            "Flatten" => Ok(Self::Flatten),
            "Floor" => Ok(Self::Floor),
            "Gather" => Ok(Self::Gather),
            "Gemm" => Ok(Self::Gemm),
            "GlobalAveragePool" => Ok(Self::GlobalAveragePool),
            "Greater" => Ok(Self::Greater),
            "HardSigmoid" => Ok(Self::HardSigmoid),
            "Hardmax" => Ok(Self::Hardmax),
            "Identity" => Ok(Self::Identity),
            "If" => Ok(Self::If),
            "IsInf" => Ok(Self::IsInf),
            "IsNaN" => Ok(Self::IsNaN),
            "LSTM" => Ok(Self::Lstm),
            "LeakyRelu" => Ok(Self::LeakyRelu),
            "Less" => Ok(Self::Less),
            "Log" => Ok(Self::Log),
            "Loop" => Ok(Self::Loop),
            "LRN" => Ok(Self::Lrn),
            "MatMul" => Ok(Self::MatMul),
            "Max" => Ok(Self::Max),
            "MaxPool" => Ok(Self::MaxPool),
            "Min" => Ok(Self::Min),
            "Mul" => Ok(Self::Mul),
            "Neg" => Ok(Self::Neg),
            "NonMaxSuppression" => Ok(Self::NonMaxSuppression),
            "NonZero" => Ok(Self::NonZero),
            "Not" => Ok(Self::Not),
            "PRelu" => Ok(Self::PRelu),
            "QLinearAdd" => Ok(Self::QLinearAdd),
            "QLinearConv" => Ok(Self::QLinearConv),
            "QLinearGlobalAveragePool" => Ok(Self::QLinearGlobalAveragePool),
            "QLinearMatMul" => Ok(Self::QLinearMatMul),
            "QuantizeLinear" => Ok(Self::QuantizeLinear),
            "Range" => Ok(Self::Range),
            "Reciprocal" => Ok(Self::Reciprocal),
            "ReduceMax" => Ok(Self::ReduceMax),
            "ReduceMean" => Ok(Self::ReduceMean),
            "ReduceMin" => Ok(Self::ReduceMin),
            "ReduceSum" => Ok(Self::ReduceSum),
            "Relu" => Ok(Self::Relu),
            "Reshape" => Ok(Self::Reshape),
            "Resize" => Ok(Self::Resize),
            "RoiAlign" => Ok(Self::RoiAlign),
            "Round" => Ok(Self::Round),
            "Scan" => Ok(Self::Scan),
            "ScatterElements" => Ok(Self::ScatterElements),
            "Selu" => Ok(Self::Selu),
            "Shape" => Ok(Self::Shape),
            "Sigmoid" => Ok(Self::Sigmoid),
            "Sign" => Ok(Self::Sign),
            "Sin" => Ok(Self::Sin),
            "Sinh" => Ok(Self::Sinh),
            "Slice" => Ok(Self::Slice),
            "Softmax" => Ok(Self::Softmax),
            "Softplus" => Ok(Self::Softplus),
            "Softsign" => Ok(Self::Softsign),
            "Split" => Ok(Self::Split),
            "Sqrt" => Ok(Self::Sqrt),
            "Squeeze" => Ok(Self::Squeeze),
            "Sub" => Ok(Self::Sub),
            "Sum" => Ok(Self::Sum),
            "Tan" => Ok(Self::Tan),
            "Tanh" => Ok(Self::Tanh),
            "ThresholdedRelu" => Ok(Self::ThresholdedRelu),
            "Tile" => Ok(Self::Tile),
            "TopK" => Ok(Self::TopK),
            "Transpose" => Ok(Self::Transpose),
            "Unsqueeze" => Ok(Self::Unsqueeze),
            "Upsample" => Ok(Self::Upsample),
            "Where" => Ok(Self::Where),
            other => Err(other.to_string()),
        }
    }

    pub fn expected_input_dtypes(self) -> &'static [Option<DType>] {
        const F: Option<DType> = Some(DType::Float);

        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => &[F, F],
            Self::Relu
            | Self::LeakyRelu
            | Self::Clip
            | Self::Sigmoid
            | Self::Exp
            | Self::Ceil
            | Self::Round
            | Self::Softmax
            | Self::Softplus
            | Self::Softsign
            | Self::Log
            | Self::Tanh
            | Self::Floor
            | Self::Sqrt
            | Self::Abs
            | Self::Hardmax
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Sinh
            | Self::Cosh
            | Self::Asinh
            | Self::Acosh
            | Self::Atanh
            | Self::Erf
            | Self::Sign
            | Self::Neg
            | Self::Reciprocal
            | Self::Elu
            | Self::Celu
            | Self::Selu
            | Self::HardSigmoid
            | Self::ThresholdedRelu
            | Self::IsNaN
            | Self::IsInf => &[F],
            Self::Conv | Self::ConvTranspose => &[F, F, F],
            Self::MatMul => &[F, F],
            Self::Gemm => &[F, F, F],
            Self::BatchNormalization => &[F, F, F, F, F],
            Self::MaxPool | Self::AveragePool | Self::GlobalAveragePool => &[F],
            Self::Resize | Self::Upsample => &[F],
            Self::DequantizeLinear => &[F, F, F],
            Self::QuantizeLinear => &[F, F, F],
            Self::NonMaxSuppression => &[F, F],
            _ => &[],
        }
    }

    pub fn infer_output_dtype(self, node: &Node, input_types: &[DType]) -> DType {
        match self {
            Self::Shape
            | Self::Less
            | Self::Equal
            | Self::Greater
            | Self::NonZero
            | Self::ArgMax
            | Self::And
            | Self::CategoryMapper
            | Self::IsNaN
            | Self::IsInf
            | Self::Not => DType::Int64,
            Self::Constant => match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => t.dtype(),
                _ => DType::Float,
            },
            Self::Cast => {
                let to = node.attrs.get_int("to").unwrap_or(1);
                crate::onnx_ir::ElemType::from_onnx(to as i32).to_dtype()
            }
            Self::Identity
            | Self::Reshape
            | Self::Squeeze
            | Self::Unsqueeze
            | Self::Flatten
            | Self::Transpose
            | Self::LayoutTranspose
            | Self::Slice
            | Self::Tile
            | Self::Gather
            | Self::Concat
            | Self::ReduceMin
            | Self::ReduceMax
            | Self::ReduceMean
            | Self::ReduceSum
            | Self::Compress
            | Self::Where
            | Self::Expand
            | Self::Range
            | Self::ScatterElements => input_types.first().copied().unwrap_or(DType::Float),
            _ => DType::Float,
        }
    }

    pub fn infer_output_shape(
        self,
        node: &Node,
        input_names: &[String],
        shape_map: &HashMap<String, crate::ShapeLayout>,
        known_values: &HashMap<String, Tensor>,
    ) -> Option<Dims> {
        let get_shape = |idx: usize| -> Option<&Dims> {
            input_names
                .get(idx)
                .filter(|s| !s.is_empty())
                .and_then(|name| shape_map.get(name))
                .map(|sl| &sl.dims)
        };
        let get_layout = |idx: usize| -> crate::Layout {
            input_names
                .get(idx)
                .filter(|s| !s.is_empty())
                .and_then(|name| shape_map.get(name))
                .map(|sl| sl.layout)
                .unwrap_or_default()
        };
        let get_value = |idx: usize| -> Option<&Tensor> {
            input_names
                .get(idx)
                .filter(|s| !s.is_empty())
                .and_then(|name| known_values.get(name))
        };

        match self {
            Self::Relu
            | Self::LeakyRelu
            | Self::Clip
            | Self::Sigmoid
            | Self::Exp
            | Self::Ceil
            | Self::Round
            | Self::Softmax
            | Self::Softplus
            | Self::Softsign
            | Self::Log
            | Self::Tanh
            | Self::Floor
            | Self::Sqrt
            | Self::Abs
            | Self::Dropout
            | Self::Hardmax
            | Self::BatchNormalization
            | Self::Identity
            | Self::Cast
            | Self::DequantizeLinear
            | Self::QuantizeLinear
            | Self::ScatterElements
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Sinh
            | Self::Cosh
            | Self::Asinh
            | Self::Acosh
            | Self::Atanh
            | Self::Erf
            | Self::Sign
            | Self::Neg
            | Self::Reciprocal
            | Self::Elu
            | Self::Celu
            | Self::Selu
            | Self::HardSigmoid
            | Self::ThresholdedRelu
            | Self::And
            | Self::IsNaN
            | Self::IsInf
            | Self::Lrn
            | Self::Not => get_shape(0).cloned(),

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Less
            | Self::Equal
            | Self::Greater
            | Self::PRelu => {
                let a = get_shape(0)?;
                let b = get_shape(1)?;
                Some(broadcast_shape(a, b))
            }

            Self::Expand => {
                let x = get_shape(0)?;
                let shape = get_value(1)?;
                let target: Vec<usize> = match shape.dtype() {
                    DType::Int64 => shape.ints().ok()?.iter().map(|&v| v as usize).collect(),
                    DType::Float => shape.floats().ok()?.iter().map(|&v| v as usize).collect(),
                    DType::String => return None,
                };
                Some(broadcast_shape(x, &target))
            }

            Self::Conv => {
                let x = get_shape(0)?;
                let w = get_shape(1)?;
                if x.len() != 4 || w.len() != 4 {
                    return None;
                }
                let layout = get_layout(0);
                let is_nhwc = layout == crate::Layout::NHWC;
                let n = x[0];
                let c_out = w[0]; // weights are always OIHW
                let (h_idx, w_idx) = if is_nhwc { (1, 2) } else { (2, 3) };
                let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();
                let strides = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
                let dilations = node
                    .attrs
                    .get_ints("dilations")
                    .unwrap_or_else(|| vec![1, 1]);
                let pads = node
                    .attrs
                    .get_ints("pads")
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);
                let ks_attr = node.attrs.get_ints("kernel_shape");

                let mut spatial_out = [0usize; 2];
                for i in 0..2 {
                    let in_dim = x[[h_idx, w_idx][i]];
                    let k = ks_attr
                        .as_ref()
                        .map(|ks| ks[i] as usize)
                        .unwrap_or(w[2 + i]);
                    let s = strides[i] as usize;
                    let d = dilations[i] as usize;
                    let ek = d * (k - 1) + 1;
                    spatial_out[i] = match auto_pad.as_str() {
                        "SAME_UPPER" | "SAME_LOWER" => in_dim.div_ceil(s),
                        "VALID" => (in_dim.saturating_sub(ek)) / s + 1,
                        _ => {
                            let p = pads[i] as usize + pads[i + 2] as usize;
                            (in_dim + p - ek) / s + 1
                        }
                    };
                }
                if is_nhwc {
                    Some(dims![n, spatial_out[0], spatial_out[1], c_out])
                } else {
                    Some(dims![n, c_out, spatial_out[0], spatial_out[1]])
                }
            }

            Self::ConvTranspose => {
                let x = get_shape(0)?;
                let w = get_shape(1)?;
                if x.len() != 4 || w.len() != 4 {
                    return None;
                }
                let n = x[0];
                let c_out = w[1]; // ConvTranspose weights are [C_in, C_out, kH, kW]
                let strides = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
                let pads = node
                    .attrs
                    .get_ints("pads")
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);
                let output_padding = node
                    .attrs
                    .get_ints("output_padding")
                    .unwrap_or_else(|| vec![0, 0]);
                let dilations = node
                    .attrs
                    .get_ints("dilations")
                    .unwrap_or_else(|| vec![1, 1]);
                let ks_attr = node.attrs.get_ints("kernel_shape");
                let group = node.attrs.get_int("group").unwrap_or(1) as usize;
                let _ = group; // used in execute, not shape
                let mut spatial_out = [0usize; 2];
                for i in 0..2 {
                    let in_dim = x[2 + i];
                    let k = ks_attr
                        .as_ref()
                        .map(|ks| ks[i] as usize)
                        .unwrap_or(w[2 + i]);
                    let s = strides[i] as usize;
                    let d = dilations[i] as usize;
                    let p = pads[i] as usize + pads[i + 2] as usize;
                    let op = output_padding[i] as usize;
                    spatial_out[i] = s * (in_dim - 1) + d * (k - 1) + 1 - p + op;
                }
                Some(dims![n, c_out, spatial_out[0], spatial_out[1]])
            }

            Self::MatMul => {
                let a = get_shape(0)?;
                let b = get_shape(1)?;
                match (a.len(), b.len()) {
                    (2, 2) => Some(dims![a[0], b[1]]),
                    (al, 2) if al >= 2 => {
                        let mut out: Dims = a[..al - 1].iter().copied().collect();
                        out.push(b[1]);
                        Some(out)
                    }
                    (1, bl) if bl >= 2 => {
                        let mut out: Dims = b[..bl - 2].iter().copied().collect();
                        out.push(b[bl - 1]);
                        Some(out)
                    }
                    (al, 1) if al >= 2 => Some(a[..al - 1].iter().copied().collect()),
                    (al, bl) if al >= 2 && bl >= 2 => {
                        let mut out = broadcast_shape(&a[..al - 2], &b[..bl - 2]);
                        out.push(a[al - 2]);
                        out.push(b[bl - 1]);
                        Some(out)
                    }
                    _ => None,
                }
            }

            Self::Gemm => {
                let a = get_shape(0)?;
                let b = get_shape(1)?;
                if a.len() != 2 || b.len() != 2 {
                    return None;
                }
                let trans_a = node.attrs.get_int("transA").unwrap_or(0) != 0;
                let trans_b = node.attrs.get_int("transB").unwrap_or(0) != 0;
                let m = if trans_a { a[1] } else { a[0] };
                let n = if trans_b { b[0] } else { b[1] };
                Some(dims![m, n])
            }

            Self::MaxPool => {
                let x = get_shape(0)?;
                if x.len() != 4 {
                    return None;
                }
                let layout = get_layout(0);
                let is_nhwc = layout == crate::Layout::NHWC;
                let (h_idx, w_idx, c_idx) = if is_nhwc { (1, 2, 3) } else { (2, 3, 1) };
                let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();
                let ks = node.attrs.get_ints("kernel_shape")?;
                let strides = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
                let pads = node
                    .attrs
                    .get_ints("pads")
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);

                let mut spatial_out = [0usize; 2];
                for i in 0..2 {
                    let in_dim = x[[h_idx, w_idx][i]];
                    let k = ks[i] as usize;
                    let s = strides[i] as usize;
                    spatial_out[i] = match auto_pad.as_str() {
                        "SAME_UPPER" | "SAME_LOWER" => in_dim.div_ceil(s),
                        "VALID" => (in_dim.saturating_sub(k)) / s + 1,
                        _ => {
                            let p = pads[i] as usize + pads[i + 2] as usize;
                            (in_dim + p - k) / s + 1
                        }
                    };
                }
                if is_nhwc {
                    Some(dims![x[0], spatial_out[0], spatial_out[1], x[c_idx]])
                } else {
                    Some(dims![x[0], x[c_idx], spatial_out[0], spatial_out[1]])
                }
            }

            Self::AveragePool => {
                let x = get_shape(0)?;
                if x.len() != 4 {
                    return None;
                }
                let layout = get_layout(0);
                let is_nhwc = layout == crate::Layout::NHWC;
                let (h_idx, w_idx, c_idx) = if is_nhwc { (1, 2, 3) } else { (2, 3, 1) };
                let auto_pad = node.attrs.get_string("auto_pad").unwrap_or_default();
                let ks = node.attrs.get_ints("kernel_shape")?;
                let strides = node.attrs.get_ints("strides").unwrap_or_else(|| vec![1, 1]);
                let pads = node
                    .attrs
                    .get_ints("pads")
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);

                let mut spatial_out = [0usize; 2];
                for i in 0..2 {
                    let in_dim = x[[h_idx, w_idx][i]];
                    let k = ks[i] as usize;
                    let s = strides[i] as usize;
                    spatial_out[i] = match auto_pad.as_str() {
                        "SAME_UPPER" | "SAME_LOWER" => in_dim.div_ceil(s),
                        "VALID" => (in_dim.saturating_sub(k)) / s + 1,
                        _ => {
                            let p = pads[i] as usize + pads[i + 2] as usize;
                            (in_dim + p - k) / s + 1
                        }
                    };
                }
                if is_nhwc {
                    Some(dims![x[0], spatial_out[0], spatial_out[1], x[c_idx]])
                } else {
                    Some(dims![x[0], x[c_idx], spatial_out[0], spatial_out[1]])
                }
            }

            Self::GlobalAveragePool => {
                let x = get_shape(0)?;
                if x.len() < 2 {
                    return None;
                }
                let layout = get_layout(0);
                let is_nhwc = layout == crate::Layout::NHWC;
                if is_nhwc && x.len() == 4 {
                    // NHWC: [N, H, W, C] → [N, 1, 1, C]
                    Some(dims![x[0], 1, 1, x[3]])
                } else {
                    // NCHW: [N, C, H, W] → [N, C, 1, 1]
                    let mut out: Dims = dims![x[0], x[1]];
                    out.resize(x.len(), 1);
                    Some(out)
                }
            }

            Self::Flatten => {
                let x = get_shape(0)?;
                let axis = node.attrs.get_int("axis").unwrap_or(1) as usize;
                let outer: usize = x[..axis].iter().product();
                let inner: usize = x[axis..].iter().product();
                Some(dims![outer, inner])
            }

            Self::Shape => {
                let x = get_shape(0)?;
                Some(dims![x.len()])
            }

            Self::Gather => {
                let data = get_shape(0)?;
                let indices = get_shape(1)?;
                let axis = node.attrs.get_int("axis").unwrap_or(0);
                let axis = if axis < 0 {
                    (data.len() as i64 + axis) as usize
                } else {
                    axis as usize
                };
                let mut out = Dims::new();
                out.extend_from_slice(&data[..axis]);
                out.extend_from_slice(indices);
                out.extend_from_slice(&data[axis + 1..]);
                Some(out)
            }

            Self::Concat => {
                let first = get_shape(0)?;
                let axis = node.attrs.get_int("axis").unwrap_or(0);
                let axis = if axis < 0 {
                    (first.len() as i64 + axis) as usize
                } else {
                    axis as usize
                };
                let mut out = first.clone();
                for i in 1..input_names.len() {
                    let s = get_shape(i)?;
                    out[axis] += s[axis];
                }
                Some(out)
            }

            Self::Transpose | Self::LayoutTranspose => {
                let x = get_shape(0)?;
                match node.attrs.get_ints("perm") {
                    Some(p) => Some(p.iter().map(|&i| x[i as usize]).collect()),
                    None => {
                        let mut out = x.clone();
                        out.reverse();
                        Some(out)
                    }
                }
            }

            Self::Reshape => {
                let x = get_shape(0)?;
                let total: usize = x.iter().product();

                let shape_vals: Vec<i64> = if let Some(t) = get_value(1) {
                    match t.dtype() {
                        DType::Int64 => t.ints().ok()?.to_vec(),
                        DType::Float => t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                        DType::String => return None,
                    }
                } else {
                    node.attrs.get_ints("shape")?
                };

                let mut dims: Dims = Dims::new();
                let mut infer_idx = None;
                for (i, &s) in shape_vals.iter().enumerate() {
                    if s == -1 {
                        infer_idx = Some(i);
                        dims.push(0);
                    } else if s == 0 {
                        dims.push(*x.get(i).unwrap_or(&1));
                    } else {
                        dims.push(s as usize);
                    }
                }
                if let Some(idx) = infer_idx {
                    let known: usize = dims
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != idx)
                        .map(|(_, &v)| v)
                        .product();
                    #[allow(clippy::arithmetic_side_effects)]
                    if known > 0 {
                        dims[idx] = total / known;
                    }
                }
                Some(dims)
            }

            Self::Squeeze => {
                let x = get_shape(0)?;
                let axes: Option<Vec<i64>> = node.attrs.get_ints("axes").or_else(|| {
                    let t = get_value(1)?;
                    Some(match t.dtype() {
                        DType::Int64 => t.ints().ok()?.to_vec(),
                        DType::Float => t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                        DType::String => return None,
                    })
                });
                match axes {
                    Some(axes) => {
                        let rank = x.len() as i64;
                        let axes: Vec<usize> = axes
                            .iter()
                            .map(|&a| {
                                if a < 0 {
                                    (rank + a) as usize
                                } else {
                                    a as usize
                                }
                            })
                            .collect();
                        Some(
                            x.iter()
                                .enumerate()
                                .filter(|(i, _)| !axes.contains(i))
                                .map(|(_, &d)| d)
                                .collect(),
                        )
                    }
                    None => Some(x.iter().copied().filter(|&d| d != 1).collect()),
                }
            }

            Self::Unsqueeze => {
                let x = get_shape(0)?;
                let axes: Vec<i64> = node.attrs.get_ints("axes").or_else(|| {
                    let t = get_value(1)?;
                    Some(match t.dtype() {
                        DType::Int64 => t.ints().ok()?.to_vec(),
                        DType::Float => t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                        DType::String => return None,
                    })
                })?;
                let out_rank = x.len() + axes.len();
                let axes_set: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (out_rank as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                let mut out = Dims::with_capacity(out_rank);
                let mut xi = 0;
                for i in 0..out_rank {
                    if axes_set.contains(&i) {
                        out.push(1);
                    } else {
                        out.push(x[xi]);
                        xi += 1;
                    }
                }
                Some(out)
            }

            Self::Slice => {
                let x = get_shape(0)?;
                let starts_t = get_value(1)?;
                let ends_t = get_value(2)?;
                let rank = x.len();

                let starts_ints: Vec<i64> = match starts_t.dtype() {
                    DType::Int64 => starts_t.ints().ok()?.to_vec(),
                    DType::Float => starts_t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                    DType::String => return None,
                };
                let ends_ints: Vec<i64> = match ends_t.dtype() {
                    DType::Int64 => ends_t.ints().ok()?.to_vec(),
                    DType::Float => ends_t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                    DType::String => return None,
                };

                let axes: Vec<usize> = if let Some(t) = get_value(3) {
                    match t.dtype() {
                        DType::Int64 => t
                            .ints()
                            .ok()?
                            .iter()
                            .map(|&a| {
                                if a < 0 {
                                    (rank as i64 + a) as usize
                                } else {
                                    a as usize
                                }
                            })
                            .collect(),
                        DType::Float => t
                            .floats()
                            .ok()?
                            .iter()
                            .map(|&a| {
                                let a = a as i64;
                                if a < 0 {
                                    (rank as i64 + a) as usize
                                } else {
                                    a as usize
                                }
                            })
                            .collect(),
                        DType::String => return None,
                    }
                } else {
                    (0..starts_ints.len()).collect()
                };

                let steps: Vec<i64> = if let Some(t) = get_value(4) {
                    match t.dtype() {
                        DType::Int64 => t.ints().ok()?.to_vec(),
                        DType::Float => t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                        DType::String => return None,
                    }
                } else {
                    vec![1; axes.len()]
                };

                let mut out = x.clone();
                for (i, &ax) in axes.iter().enumerate() {
                    let dim = x[ax] as i64;
                    let step = steps[i];
                    let mut s = starts_ints[i];
                    let mut e = ends_ints[i];
                    if step > 0 {
                        if s < 0 {
                            s += dim;
                        }
                        if e < 0 {
                            e += dim;
                        }
                        s = s.clamp(0, dim);
                        e = e.clamp(0, dim);
                        out[ax] = if e <= s {
                            0
                        } else {
                            ((e - s - 1) / step + 1) as usize
                        };
                    } else {
                        if s < 0 {
                            s += dim;
                        }
                        if e < -dim {
                            e = -1;
                        } else if e < 0 {
                            e += dim;
                        }
                        s = s.clamp(-1, dim - 1);
                        e = e.clamp(-1, dim - 1);
                        out[ax] = if s <= e {
                            0
                        } else {
                            ((s - e - 1) / (-step) + 1) as usize
                        };
                    }
                }
                Some(out)
            }

            Self::Tile => {
                let x = get_shape(0)?;
                let repeats = get_value(1)?;
                let reps: Vec<usize> = match repeats.dtype() {
                    DType::Int64 => repeats.ints().ok()?.iter().map(|&v| v as usize).collect(),
                    DType::Float => repeats.floats().ok()?.iter().map(|&v| v as usize).collect(),
                    DType::String => return None,
                };
                Some(x.iter().zip(reps.iter()).map(|(&d, &r)| d * r).collect())
            }

            Self::Resize => {
                let x = get_shape(0)?;
                if let Some(sizes) = get_value(3) {
                    if sizes.numel() > 0 {
                        return Some(match sizes.dtype() {
                            DType::Int64 => {
                                sizes.ints().ok()?.iter().map(|&v| v as usize).collect()
                            }
                            DType::Float => {
                                sizes.floats().ok()?.iter().map(|&v| v as usize).collect()
                            }
                            DType::String => return None,
                        });
                    }
                }
                if let Some(scales) = get_value(2) {
                    if scales.numel() > 0 {
                        let sf = scales.floats().ok()?;
                        return Some(
                            x.iter()
                                .zip(sf.iter())
                                .map(|(&d, &s)| (d as f32 * s) as usize)
                                .collect(),
                        );
                    }
                }
                None
            }

            Self::Upsample => {
                let x = get_shape(0)?;
                let scales = get_value(1)?;
                let sf = scales.floats().ok()?;
                Some(
                    x.iter()
                        .zip(sf.iter())
                        .map(|(&d, &s)| (d as f32 * s) as usize)
                        .collect(),
                )
            }

            Self::ReduceMin => {
                let x = get_shape(0)?;
                let keepdims = node.attrs.get_int("keepdims").unwrap_or(1) != 0;
                let axes: Vec<i64> = node.attrs.get_ints("axes").or_else(|| {
                    let t = get_value(1)?;
                    Some(match t.dtype() {
                        DType::Int64 => t.ints().ok()?.to_vec(),
                        DType::Float => t.floats().ok()?.iter().map(|&v| v as i64).collect(),
                        DType::String => return None,
                    })
                })?;
                let rank = x.len() as i64;
                let axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (rank + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                if keepdims {
                    Some(
                        x.iter()
                            .enumerate()
                            .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                            .collect(),
                    )
                } else {
                    Some(
                        x.iter()
                            .enumerate()
                            .filter(|(i, _)| !axes.contains(i))
                            .map(|(_, &d)| d)
                            .collect(),
                    )
                }
            }

            Self::Constant => match node.attrs.get("value") {
                Some(Attr::Tensor(t)) => Some(t.dims.clone()),
                _ => None,
            },

            Self::ArgMax => {
                let x = get_shape(0)?;
                let axis_raw = node.attrs.get_int("axis").unwrap_or(0);
                let axis = if axis_raw < 0 {
                    (x.len() as i64 + axis_raw) as usize
                } else {
                    axis_raw as usize
                };
                let keepdims = node.attrs.get_int("keepdims").unwrap_or(1) != 0;
                let mut out = Dims::new();
                for (i, &d) in x.iter().enumerate() {
                    if i == axis {
                        if keepdims {
                            out.push(1);
                        }
                    } else {
                        out.push(d);
                    }
                }
                if out.is_empty() {
                    out.push(1);
                }
                Some(out)
            }

            Self::CategoryMapper => get_shape(0).cloned(),

            Self::Sum => get_shape(0).cloned(),

            Self::Where => {
                let x = get_shape(1)?;
                let y = get_shape(2)?;
                Some(broadcast_shape(x, y))
            }

            Self::ReduceMax | Self::ReduceMean | Self::ReduceSum => {
                let x = get_shape(0)?;
                let keepdims = node.attrs.get_int("keepdims").unwrap_or(1) != 0;
                let axes: Option<Vec<i64>> = node.attrs.get_ints("axes");
                if let Some(axes) = axes {
                    let rank = x.len() as i64;
                    let axes: Vec<usize> = axes
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (rank + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect();
                    if keepdims {
                        Some(
                            x.iter()
                                .enumerate()
                                .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                                .collect(),
                        )
                    } else {
                        Some(
                            x.iter()
                                .enumerate()
                                .filter(|(i, _)| !axes.contains(i))
                                .map(|(_, &d)| d)
                                .collect(),
                        )
                    }
                } else {
                    None
                }
            }

            Self::NonMaxSuppression
            | Self::Loop
            | Self::Split
            | Self::If
            | Self::NonZero
            | Self::TopK
            | Self::Range
            | Self::Max
            | Self::Min
            | Self::ConstantOfShape
            | Self::RoiAlign
            | Self::Lstm
            | Self::Scan
            | Self::Compress => None,

            Self::QLinearConv
            | Self::QLinearAdd
            | Self::QLinearMatMul
            | Self::QLinearGlobalAveragePool => None,
        }
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
