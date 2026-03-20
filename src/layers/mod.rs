use anyhow::Context;
pub mod abs;
pub mod add;
pub mod argmax;
pub mod auto_cast;
pub mod batch_norm;
pub mod cast;
pub mod category_mapper;
pub mod ceil;
pub mod clip;
pub mod compress;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod conv;
pub mod dequantize_linear;
pub mod div;
pub mod dropout;
pub mod equal;
pub mod exp;
pub mod expand;
pub mod flatten;
pub mod floor;
pub mod gather;
pub mod gemm;
pub mod global_avg_pool;
pub mod greater;
pub mod hardmax;
pub mod identity;
pub mod if_op;
pub mod leaky_relu;
pub mod less;
pub mod log;
pub mod loop_op;
pub mod lstm;
pub mod matmul;
pub mod max_op;
pub mod maxpool;
pub mod min_op;
pub mod mul;
pub mod nms;
pub mod nonzero;
pub mod op_type;
pub mod plan;
pub mod qlinear_add;
pub mod qlinear_conv;
pub mod qlinear_global_avg_pool;
pub mod qlinear_matmul;
pub mod quantize_linear;
pub mod range;
pub mod reduce_max;
pub mod reduce_min;
pub mod reduce_sum;
pub mod relu;
pub mod reshape;
pub mod resize;
pub mod roi_align;
pub mod round;
pub mod scan;
pub mod scatter_elements;
pub mod shape_op;
pub mod sigmoid;
pub mod slice;
pub mod softmax;
pub mod softplus;
pub mod split;
pub mod sqrt;
pub mod squeeze;
pub mod sub;
pub mod sum;
pub mod tanh;
pub mod tile;
pub mod topk;
pub mod transpose;
pub mod unary_ops;
pub mod unsqueeze;
pub mod where_op;
#[cfg(feature = "xnnpack")]
pub mod xnnpack_subgraph;

use std::collections::HashMap;

pub use op_type::OpType;
pub use plan::Plan;
pub use plan::PlanNode;
pub use plan::build_node;
pub use plan::execute_node;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;

pub trait Layer: Send {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()>;
}

pub fn dequantize_into(data: &[f32], scale: f32, zero_point: f32, out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(data.iter()) {
        *o = (v - zero_point) * scale;
    }
}

pub fn quantize_u8_into(data: &[f32], scale: f32, zero_point: f32, out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(data.iter()) {
        *o = (v / scale + zero_point).round().clamp(0.0, 255.0);
    }
}

pub fn binary_op(
    a: &Tensor,
    b: &Tensor,
    output: &mut Tensor,
    legacy_broadcast: bool,
    axis: usize,
    op: fn(f32, f32) -> f32,
) -> Result<()> {
    // Int64 fast path: cast to float, apply op, cast result back to int64
    if a.dtype() == crate::DType::Int64 || b.dtype() == crate::DType::Int64 {
        return binary_op_i64(a, b, output, legacy_broadcast, axis, op);
    }

    let mut b_dims_buf = [1usize; 8];
    let b_dims_len = if legacy_broadcast && b.dims.len() < a.dims.len() {
        for (i, &d) in b.dims.iter().enumerate() {
            b_dims_buf[axis + i] = d;
        }
        a.dims.len()
    } else {
        for (i, &d) in b.dims.iter().enumerate() {
            b_dims_buf[i] = d;
        }
        b.dims.len()
    };
    let b_dims = &b_dims_buf[..b_dims_len];

    let ndim = a.dims.len().max(b_dims_len);
    let mut out_shape = [0usize; 8];
    broadcast_shape_into(&a.dims, b_dims, &mut out_shape[..ndim]);
    let out_dims = &out_shape[..ndim];
    let numel: usize = out_dims.iter().product();
    let buf = output.as_mut_f32(numel);

    let a_f = a.floats().context("in binary_op")?;
    let b_f = b.floats().context("in binary_op")?;

    // Fast path: identical shapes — no broadcast needed
    if a.dims.as_slice() == b_dims {
        for i in 0..numel {
            buf[i] = op(a_f[i], b_f[i]);
        }
        output.set_dims(out_dims);
        return Ok(());
    }

    // Fast path: b is scalar
    if b_f.len() == 1 {
        let bv = b_f[0];
        for i in 0..numel {
            buf[i] = op(a_f[i], bv);
        }
        output.set_dims(out_dims);
        return Ok(());
    }

    // Fast path: a is scalar
    if a_f.len() == 1 {
        let av = a_f[0];
        for i in 0..numel {
            buf[i] = op(av, b_f[i]);
        }
        output.set_dims(out_dims);
        return Ok(());
    }

    // Fast path: per-channel broadcast where b has one non-1 dim matching a
    // Covers common patterns like [N,C,H,W] + [1,C,1,1] or [N,C,H,W] + [C]
    if a_f.len() == numel && b_f.len() > 1 && b_f.len() < numel {
        // Compute b's strides into the output space
        let b_rank = b_dims.len();
        let offset = ndim - b_rank;
        // Build a stride multiplier for b indexing
        let mut inner = 1usize;
        let mut b_stride = 0usize;
        let mut single_axis = true;
        let mut non1_count = 0usize;
        for i in (0..b_rank).rev() {
            if b_dims[i] != 1 {
                non1_count += 1;
                if non1_count == 1 {
                    b_stride = inner;
                } else {
                    single_axis = false;
                    break;
                }
            }
            inner *= out_dims[i + offset];
        }
        if single_axis && non1_count == 1 {
            let b_len = b_f.len();
            for i in 0..numel {
                let bi = (i / b_stride) % b_len;
                buf[i] = op(a_f[i], b_f[bi]);
            }
            output.set_dims(out_dims);
            return Ok(());
        }
    }

    // General fallback with per-element broadcast index
    let mut index = [0usize; 8];
    for val in buf.iter_mut() {
        let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
        let bi = broadcast_index(&index[..ndim], b_dims, out_dims);
        *val = op(a_f[ai], b_f[bi]);

        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < out_dims[d] {
                break;
            }
            index[d] = 0;
        }
    }

    output.set_dims(out_dims);
    Ok(())
}

fn binary_op_i64(
    a: &Tensor,
    b: &Tensor,
    output: &mut Tensor,
    _legacy_broadcast: bool,
    _axis: usize,
    op: fn(f32, f32) -> f32,
) -> Result<()> {
    let a_vals: Vec<f32> = match a.dtype() {
        crate::DType::Int64 => a.ints().context("in binary_op")?.iter().map(|&v| v as f32).collect(),
        _ => a.floats().context("in binary_op")?.to_vec(),
    };
    let b_vals: Vec<f32> = match b.dtype() {
        crate::DType::Int64 => b.ints().context("in binary_op")?.iter().map(|&v| v as f32).collect(),
        _ => b.floats().context("in binary_op")?.to_vec(),
    };
    let ndim = a.dims.len().max(b.dims.len());
    let mut out_shape = [0usize; 8];
    broadcast_shape_into(&a.dims, &b.dims, &mut out_shape[..ndim]);
    let out_dims = &out_shape[..ndim];
    let numel: usize = out_dims.iter().product();

    // Determine output type: int64 if both inputs are int64
    let out_is_int = a.dtype() == crate::DType::Int64 && b.dtype() == crate::DType::Int64;

    if out_is_int {
        let buf = output.as_mut_i64(numel);
        if a.dims.as_slice() == b.dims.as_slice() {
            for i in 0..numel {
                buf[i] = op(a_vals[i], b_vals[i]) as i64;
            }
        } else if b_vals.len() == 1 {
            let bv = b_vals[0];
            for i in 0..numel {
                buf[i] = op(a_vals[i], bv) as i64;
            }
        } else if a_vals.len() == 1 {
            let av = a_vals[0];
            for i in 0..numel {
                buf[i] = op(av, b_vals[i]) as i64;
            }
        } else {
            let mut index = [0usize; 8];
            for val in buf.iter_mut() {
                let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
                let bi = broadcast_index(&index[..ndim], &b.dims, out_dims);
                *val = op(a_vals[ai], b_vals[bi]) as i64;
                for d in (0..ndim).rev() {
                    index[d] += 1;
                    if index[d] < out_dims[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
        }
    } else {
        let buf = output.as_mut_f32(numel);
        if a.dims.as_slice() == b.dims.as_slice() {
            for i in 0..numel {
                buf[i] = op(a_vals[i], b_vals[i]);
            }
        } else if b_vals.len() == 1 {
            let bv = b_vals[0];
            for i in 0..numel {
                buf[i] = op(a_vals[i], bv);
            }
        } else if a_vals.len() == 1 {
            let av = a_vals[0];
            for i in 0..numel {
                buf[i] = op(av, b_vals[i]);
            }
        } else {
            let mut index = [0usize; 8];
            for val in buf.iter_mut() {
                let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
                let bi = broadcast_index(&index[..ndim], &b.dims, out_dims);
                *val = op(a_vals[ai], b_vals[bi]);
                for d in (0..ndim).rev() {
                    index[d] += 1;
                    if index[d] < out_dims[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
        }
    }
    output.set_dims(out_dims);
    Ok(())
}
