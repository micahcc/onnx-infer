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
use crate::broadcast_shape;

pub use op_type::OpType;
pub use plan::build_plan;
pub use plan::build_node;
pub use plan::execute_node;

pub trait Layer {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()>;
}

pub enum PlanNode {
    Single {
        output: String,
        layer: Box<dyn Layer>,
    },
    Loop(loop_op::Loop),
}

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
