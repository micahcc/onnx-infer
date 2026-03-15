pub mod add;
pub mod auto_cast;
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

use crate::DType;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Add,
    BatchNormalization,
    Cast,
    Ceil,
    Clip,
    Concat,
    Constant,
    Conv,
    DequantizeLinear,
    Div,
    Exp,
    Flatten,
    Gather,
    Gemm,
    GlobalAveragePool,
    Identity,
    LeakyRelu,
    Loop,
    MatMul,
    MaxPool,
    Mul,
    NonMaxSuppression,
    QLinearAdd,
    QLinearConv,
    QLinearGlobalAveragePool,
    QLinearMatMul,
    QuantizeLinear,
    ReduceMin,
    Relu,
    Reshape,
    Resize,
    Round,
    Shape,
    Sigmoid,
    Slice,
    Softmax,
    Squeeze,
    Sub,
    Tile,
    Transpose,
    Unsqueeze,
}

impl OpType {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "Add" => Ok(Self::Add),
            "BatchNormalization" => Ok(Self::BatchNormalization),
            "Cast" => Ok(Self::Cast),
            "Ceil" => Ok(Self::Ceil),
            "Clip" => Ok(Self::Clip),
            "Concat" => Ok(Self::Concat),
            "Constant" => Ok(Self::Constant),
            "Conv" => Ok(Self::Conv),
            "DequantizeLinear" => Ok(Self::DequantizeLinear),
            "Div" => Ok(Self::Div),
            "Exp" => Ok(Self::Exp),
            "Flatten" => Ok(Self::Flatten),
            "Gather" => Ok(Self::Gather),
            "Gemm" => Ok(Self::Gemm),
            "GlobalAveragePool" => Ok(Self::GlobalAveragePool),
            "Identity" => Ok(Self::Identity),
            "LeakyRelu" => Ok(Self::LeakyRelu),
            "Loop" => Ok(Self::Loop),
            "MatMul" => Ok(Self::MatMul),
            "MaxPool" => Ok(Self::MaxPool),
            "Mul" => Ok(Self::Mul),
            "NonMaxSuppression" => Ok(Self::NonMaxSuppression),
            "QLinearAdd" => Ok(Self::QLinearAdd),
            "QLinearConv" => Ok(Self::QLinearConv),
            "QLinearGlobalAveragePool" => Ok(Self::QLinearGlobalAveragePool),
            "QLinearMatMul" => Ok(Self::QLinearMatMul),
            "QuantizeLinear" => Ok(Self::QuantizeLinear),
            "ReduceMin" => Ok(Self::ReduceMin),
            "Relu" => Ok(Self::Relu),
            "Reshape" => Ok(Self::Reshape),
            "Resize" => Ok(Self::Resize),
            "Round" => Ok(Self::Round),
            "Shape" => Ok(Self::Shape),
            "Sigmoid" => Ok(Self::Sigmoid),
            "Slice" => Ok(Self::Slice),
            "Softmax" => Ok(Self::Softmax),
            "Squeeze" => Ok(Self::Squeeze),
            "Sub" => Ok(Self::Sub),
            "Tile" => Ok(Self::Tile),
            "Transpose" => Ok(Self::Transpose),
            "Unsqueeze" => Ok(Self::Unsqueeze),
            other => Err(other.to_string()),
        }
    }

    pub fn expected_input_dtypes(self) -> &'static [Option<DType>] {
        const F: Option<DType> = Some(DType::Float);

        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => &[F, F],
            Self::Relu | Self::LeakyRelu | Self::Clip | Self::Sigmoid | Self::Exp | Self::Ceil
            | Self::Round | Self::Softmax => &[F],
            Self::Conv => &[F, F, F],
            Self::MatMul => &[F, F],
            Self::Gemm => &[F, F, F],
            Self::BatchNormalization => &[F, F, F, F, F],
            Self::MaxPool | Self::GlobalAveragePool => &[F],
            Self::Resize => &[F],
            Self::DequantizeLinear => &[F, F, F],
            Self::QuantizeLinear => &[F, F, F],
            Self::NonMaxSuppression => &[F, F],
            _ => &[],
        }
    }

    pub fn infer_output_dtype(self, node: &NodeProto, input_types: &[DType]) -> DType {
        match self {
            Self::Shape => DType::Int64,
            Self::Constant => node
                .attribute
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
                .map(|t| {
                    if t.data_type == 6 || t.data_type == 7 {
                        DType::Int64
                    } else {
                        DType::Float
                    }
                })
                .unwrap_or(DType::Float),
            Self::Cast => {
                let to = get_attr_int(node, "to").unwrap_or(1);
                if to == 6 || to == 7 {
                    DType::Int64
                } else {
                    DType::Float
                }
            }
            Self::Identity | Self::Reshape | Self::Squeeze | Self::Unsqueeze | Self::Flatten
            | Self::Transpose | Self::Slice | Self::Tile | Self::Gather | Self::Concat
            | Self::ReduceMin => input_types.first().copied().unwrap_or(DType::Float),
            _ => DType::Float,
        }
    }

    pub fn infer_output_shape(
        self,
        node: &NodeProto,
        input_names: &[String],
        shape_map: &HashMap<String, Vec<usize>>,
        known_values: &HashMap<String, Tensor>,
    ) -> Option<Vec<usize>> {
        let get_shape = |idx: usize| -> Option<&Vec<usize>> {
            input_names
                .get(idx)
                .filter(|s| !s.is_empty())
                .and_then(|name| shape_map.get(name))
        };
        let get_value = |idx: usize| -> Option<&Tensor> {
            input_names
                .get(idx)
                .filter(|s| !s.is_empty())
                .and_then(|name| known_values.get(name))
        };

        match self {
            // Same shape as first input
            Self::Relu | Self::LeakyRelu | Self::Clip | Self::Sigmoid | Self::Exp | Self::Ceil
            | Self::Round | Self::Softmax | Self::BatchNormalization | Self::Identity
            | Self::Cast | Self::DequantizeLinear | Self::QuantizeLinear => get_shape(0).cloned(),

            // Binary broadcast
            Self::Add | Self::Sub | Self::Mul | Self::Div => {
                let a = get_shape(0)?;
                let b = get_shape(1)?;
                Some(broadcast_shape(a, b))
            }

            Self::Conv => {
                let x = get_shape(0)?;
                let w = get_shape(1)?;
                if x.len() != 4 || w.len() != 4 {
                    return None;
                }
                let n = x[0];
                let c_out = w[0];
                let auto_pad = get_attr_string(node, "auto_pad").unwrap_or_default();
                let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
                let dilations = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
                let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
                let ks_attr = get_attr_ints(node, "kernel_shape");

                let mut out_dims = vec![n, c_out];
                for i in 0..2 {
                    let in_dim = x[2 + i];
                    let k = ks_attr
                        .as_ref()
                        .map(|ks| ks[i] as usize)
                        .unwrap_or(w[2 + i]);
                    let s = strides[i] as usize;
                    let d = dilations[i] as usize;
                    let ek = d * (k - 1) + 1;
                    let out_dim = match auto_pad.as_str() {
                        "SAME_UPPER" | "SAME_LOWER" => (in_dim + s - 1) / s,
                        "VALID" => (in_dim.saturating_sub(ek)) / s + 1,
                        _ => {
                            let p = pads[i] as usize + pads[i + 2] as usize;
                            (in_dim + p - ek) / s + 1
                        }
                    };
                    out_dims.push(out_dim);
                }
                Some(out_dims)
            }

            Self::MatMul => {
                let a = get_shape(0)?;
                let b = get_shape(1)?;
                match (a.len(), b.len()) {
                    (2, 2) => Some(vec![a[0], b[1]]),
                    (al, 2) if al >= 2 => {
                        let mut out = a[..al - 1].to_vec();
                        out.push(b[1]);
                        Some(out)
                    }
                    (1, bl) if bl >= 2 => {
                        let mut out = b[..bl - 2].to_vec();
                        out.push(b[bl - 1]);
                        Some(out)
                    }
                    (al, 1) if al >= 2 => Some(a[..al - 1].to_vec()),
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
                let trans_a = get_attr_int(node, "transA").unwrap_or(0) != 0;
                let trans_b = get_attr_int(node, "transB").unwrap_or(0) != 0;
                let m = if trans_a { a[1] } else { a[0] };
                let n = if trans_b { b[0] } else { b[1] };
                Some(vec![m, n])
            }

            Self::MaxPool => {
                let x = get_shape(0)?;
                if x.len() != 4 {
                    return None;
                }
                let auto_pad = get_attr_string(node, "auto_pad").unwrap_or_default();
                let ks = get_attr_ints(node, "kernel_shape")?;
                let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
                let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

                let mut out_dims = vec![x[0], x[1]];
                for i in 0..2 {
                    let in_dim = x[2 + i];
                    let k = ks[i] as usize;
                    let s = strides[i] as usize;
                    let out_dim = match auto_pad.as_str() {
                        "SAME_UPPER" | "SAME_LOWER" => (in_dim + s - 1) / s,
                        "VALID" => (in_dim.saturating_sub(k)) / s + 1,
                        _ => {
                            let p = pads[i] as usize + pads[i + 2] as usize;
                            (in_dim + p - k) / s + 1
                        }
                    };
                    out_dims.push(out_dim);
                }
                Some(out_dims)
            }

            Self::GlobalAveragePool => {
                let x = get_shape(0)?;
                if x.len() < 2 {
                    return None;
                }
                let mut out = vec![x[0], x[1]];
                for _ in 2..x.len() {
                    out.push(1);
                }
                Some(out)
            }

            Self::Flatten => {
                let x = get_shape(0)?;
                let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;
                let outer: usize = x[..axis].iter().product();
                let inner: usize = x[axis..].iter().product();
                Some(vec![outer, inner])
            }

            Self::Shape => {
                let x = get_shape(0)?;
                Some(vec![x.len()])
            }

            Self::Gather => {
                let data = get_shape(0)?;
                let indices = get_shape(1)?;
                let axis = get_attr_int(node, "axis").unwrap_or(0);
                let axis = if axis < 0 {
                    (data.len() as i64 + axis) as usize
                } else {
                    axis as usize
                };
                let mut out = Vec::new();
                out.extend_from_slice(&data[..axis]);
                out.extend_from_slice(indices);
                out.extend_from_slice(&data[axis + 1..]);
                Some(out)
            }

            Self::Concat => {
                let first = get_shape(0)?;
                let axis = get_attr_int(node, "axis").unwrap_or(0);
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

            Self::Transpose => {
                let x = get_shape(0)?;
                match get_attr_ints(node, "perm") {
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

                // Try shape from known value (input 1), then from attribute
                let shape_vals: Vec<i64> = if let Some(t) = get_value(1) {
                    match t.dtype {
                        DType::Int64 => t.ints().to_vec(),
                        DType::Float => t.floats().iter().map(|&v| v as i64).collect(),
                    }
                } else if let Some(attr) = get_attr_ints(node, "shape") {
                    attr
                } else {
                    return None;
                };

                let mut dims: Vec<usize> = Vec::new();
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
                    if known > 0 {
                        dims[idx] = total / known;
                    }
                }
                Some(dims)
            }

            Self::Squeeze => {
                let x = get_shape(0)?;
                let axes: Option<Vec<i64>> = get_attr_ints(node, "axes").or_else(|| {
                    let t = get_value(1)?;
                    Some(match t.dtype {
                        DType::Int64 => t.ints().to_vec(),
                        DType::Float => t.floats().iter().map(|&v| v as i64).collect(),
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
                let axes: Vec<i64> = get_attr_ints(node, "axes").or_else(|| {
                    let t = get_value(1)?;
                    Some(match t.dtype {
                        DType::Int64 => t.ints().to_vec(),
                        DType::Float => t.floats().iter().map(|&v| v as i64).collect(),
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
                let mut out = Vec::with_capacity(out_rank);
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

                let starts_ints: Vec<i64> = match starts_t.dtype {
                    DType::Int64 => starts_t.ints().to_vec(),
                    DType::Float => starts_t.floats().iter().map(|&v| v as i64).collect(),
                };
                let ends_ints: Vec<i64> = match ends_t.dtype {
                    DType::Int64 => ends_t.ints().to_vec(),
                    DType::Float => ends_t.floats().iter().map(|&v| v as i64).collect(),
                };

                let axes: Vec<usize> = if let Some(t) = get_value(3) {
                    match t.dtype {
                        DType::Int64 => t
                            .ints()
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
                    }
                } else {
                    (0..starts_ints.len()).collect()
                };

                let steps: Vec<i64> = if let Some(t) = get_value(4) {
                    match t.dtype {
                        DType::Int64 => t.ints().to_vec(),
                        DType::Float => t.floats().iter().map(|&v| v as i64).collect(),
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
                let reps: Vec<usize> = match repeats.dtype {
                    DType::Int64 => repeats.ints().iter().map(|&v| v as usize).collect(),
                    DType::Float => repeats.floats().iter().map(|&v| v as usize).collect(),
                };
                Some(x.iter().zip(reps.iter()).map(|(&d, &r)| d * r).collect())
            }

            Self::Resize => {
                let x = get_shape(0)?;
                // Try sizes (input 3)
                if let Some(sizes) = get_value(3) {
                    if sizes.numel() > 0 {
                        return Some(match sizes.dtype {
                            DType::Int64 => sizes.ints().iter().map(|&v| v as usize).collect(),
                            DType::Float => sizes.floats().iter().map(|&v| v as usize).collect(),
                        });
                    }
                }
                // Try scales (input 2)
                if let Some(scales) = get_value(2) {
                    if scales.numel() > 0 {
                        let sf = scales.floats();
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

            Self::ReduceMin => {
                let x = get_shape(0)?;
                let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) != 0;
                let axes: Vec<i64> = get_attr_ints(node, "axes").or_else(|| {
                    let t = get_value(1)?;
                    Some(match t.dtype {
                        DType::Int64 => t.ints().to_vec(),
                        DType::Float => t.floats().iter().map(|&v| v as i64).collect(),
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

            Self::Constant => {
                let attr = node.attribute.iter().find(|a| a.name == "value")?;
                let t = attr.t.as_ref()?;
                Some(t.dims.iter().map(|&d| d as usize).collect())
            }

            Self::NonMaxSuppression | Self::Loop => None,

            Self::QLinearConv | Self::QLinearAdd | Self::QLinearMatMul
            | Self::QLinearGlobalAveragePool => None,
        }
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
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

pub enum PlanNode {
    Single {
        output: String,
        layer: Box<dyn Layer>,
    },
    Loop(loop_op::Loop),
}

/// Try to compute a tensor value at build time for shape-computation ops.
/// Uses known constant values and the shape map (for Shape op).
fn try_propagate_value(
    op: OpType,
    node: &NodeProto,
    input_names: &[String],
    known_values: &HashMap<String, Tensor>,
    initializers: &HashMap<String, Tensor>,
    shape_map: &HashMap<String, Vec<usize>>,
) -> Option<Tensor> {
    // Shape produces a value from input shape, not input value
    if op == OpType::Shape {
        let name = input_names.first().filter(|s| !s.is_empty())?;
        let shape = shape_map.get(name)?;
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        return Some(Tensor::new_i64(vec![dims.len()], dims));
    }

    // Constant: extract directly from node attribute
    if op == OpType::Constant {
        let attr = node.attribute.iter().find(|a| a.name == "value")?;
        return Tensor::from_proto(attr.t.as_ref()?).ok();
    }

    // Only propagate through ops commonly used in shape computation
    match op {
        OpType::Gather | OpType::Unsqueeze | OpType::Squeeze | OpType::Concat | OpType::Cast
        | OpType::Identity | OpType::Reshape | OpType::Flatten | OpType::Slice => {}
        _ => return None,
    }

    // Collect known inputs into a temp map, bail if any are missing
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

/// Build the execution plan from an ONNX graph.
/// Inserts automatic cast nodes and pre-computes tensor shapes.
pub fn build_plan(
    graph: &crate::onnx::GraphProto,
    input_sizes: &HashMap<String, Vec<usize>>,
) -> Result<(
    Vec<PlanNode>,
    HashMap<String, Tensor>,
    Vec<String>,
    HashMap<String, Vec<usize>>,
)> {
    let mut initializers = HashMap::new();
    for init in &graph.initializer {
        if !init.name.is_empty() {
            initializers.insert(init.name.clone(), Tensor::from_proto(init)?);
        }
    }

    let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

    // Type map: tensor name → DType
    let mut type_map: HashMap<String, DType> = HashMap::new();
    for (name, tensor) in &initializers {
        type_map.insert(name.clone(), tensor.dtype);
    }
    for input in &graph.input {
        if !type_map.contains_key(&input.name) {
            let dtype = input
                .r#type
                .as_ref()
                .and_then(|t| t.value.as_ref())
                .map(|v| match v {
                    crate::onnx::type_proto::Value::TensorType(tt) => {
                        if tt.elem_type == 6 || tt.elem_type == 7 {
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

    // Shape map: tensor name → dimensions
    let mut shape_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (name, tensor) in &initializers {
        shape_map.insert(name.clone(), tensor.dims.clone());
    }
    for (name, dims) in input_sizes {
        shape_map.insert(name.clone(), dims.clone());
    }

    // Known values: tensors whose values are statically known (for shape computation)
    let mut known_values: HashMap<String, Tensor> = HashMap::new();

    let mut plan = Vec::new();
    let mut cast_counter = 0usize;

    for node in &graph.node {
        let op = OpType::parse(&node.op_type)
            .map_err(|s| InferenceError::UnsupportedOperator(s))?;

        let expected = op.expected_input_dtypes();
        let mut modified_inputs = node.input.clone();

        // Collect actual input types for output type inference
        let mut input_types = Vec::new();
        for name in &node.input {
            if let Some(&dt) = type_map.get(name) {
                input_types.push(dt);
            }
        }

        // Insert cast nodes where needed
        for (i, input_name) in node.input.iter().enumerate() {
            if input_name.is_empty() {
                continue;
            }
            if let Some(Some(expected_dt)) = expected.get(i) {
                if let Some(&actual_dt) = type_map.get(input_name) {
                    if actual_dt != *expected_dt {
                        let cast_name = format!("__auto_cast_{cast_counter}__");
                        cast_counter += 1;
                        plan.push(PlanNode::Single {
                            output: cast_name.clone(),
                            layer: Box::new(auto_cast::AutoCastF32::new(input_name.clone())),
                        });
                        type_map.insert(cast_name.clone(), DType::Float);
                        modified_inputs[i] = cast_name;
                    }
                }
            }
        }

        // Record output type
        let out_dtype = op.infer_output_dtype(node, &input_types);
        let out_name = node.output.first().filter(|s| !s.is_empty());
        if let Some(out_name) = out_name {
            type_map.insert(out_name.clone(), out_dtype);
        }

        // Try to propagate known values (constant folding for shape computation)
        if let Some(tensor) =
            try_propagate_value(op, node, &node.input, &known_values, &initializers, &shape_map)
        {
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

        plan.push(build_node(op, node, modified_inputs)?);
    }

    Ok((plan, initializers, output_names, shape_map))
}

fn build_node(op: OpType, node: &NodeProto, inputs: Vec<String>) -> Result<PlanNode> {
    if op == OpType::Loop {
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
        OpType::Resize => Box::new(resize::Resize::new(inputs)),
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
        OpType::Loop => unreachable!(),
    };

    Ok(PlanNode::Single { output, layer })
}

/// Execute a single ONNX node using the dispatch mechanism.
/// Used by Loop's body execution which still works with raw NodeProto.
pub fn execute_node(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let op = OpType::parse(&node.op_type)
        .map_err(|s| InferenceError::UnsupportedOperator(s))?;

    let _span = tracing::trace_span!("op", op = %op, name = %node.name).entered();

    if op == OpType::Loop {
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

    // Auto-cast inputs where the layer expects Float but got Int64
    let expected = op.expected_input_dtypes();
    let mut to_cast: Vec<(usize, String)> = Vec::new();
    for (i, input_name) in node.input.iter().enumerate() {
        if input_name.is_empty() {
            continue;
        }
        if let Some(Some(expected_dt)) = expected.get(i) {
            if let Some(tensor) = values.get(input_name) {
                if tensor.dtype != *expected_dt {
                    to_cast.push((i, input_name.clone()));
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

    let mut plan_node = build_node(op, node, modified_inputs)?;
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
