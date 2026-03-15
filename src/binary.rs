use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape;
use crate::get_attr_int;
use crate::get_tensor;
use crate::onnx::NodeProto;

pub fn exec_binary_op(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
    op: fn(f32, f32) -> f32,
) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let b = get_tensor(values, &node.input[1])?;

    // Handle legacy ONNX broadcast attribute
    let legacy_broadcast = get_attr_int(node, "broadcast").unwrap_or(0);
    let b_dims = if legacy_broadcast == 1 && b.dims.len() < a.dims.len() {
        let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
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

    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];

    for val in buf.iter_mut() {
        let ai = broadcast_index(&index, &a.dims, &out_shape);
        let bi = broadcast_index(&index, &b_dims, &out_shape);
        *val = op(a.f32_at(ai), b.f32_at(bi));

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

pub fn exec_div(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    exec_binary_op(node, values, output, |a, b| a / b)
}

pub fn exec_sub(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    exec_binary_op(node, values, output, |a, b| a - b)
}

pub fn exec_mul(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    exec_binary_op(node, values, output, |a, b| a * b)
}

pub fn exec_add(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    exec_binary_op(node, values, output, |a, b| a + b)
}
