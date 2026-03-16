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
pub mod op_type;
pub mod plan;
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

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;

pub use op_type::OpType;
pub use plan::Plan;
pub use plan::PlanNode;
pub use plan::build_node;
pub use plan::execute_node;

pub trait Layer {
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

    let a_f = a.floats();
    let b_f = b.floats();
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
