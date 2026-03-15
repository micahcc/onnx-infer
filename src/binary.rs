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
    values: &mut HashMap<String, Tensor>,
    op: fn(f32, f32) -> f32,
) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let mut b = get_tensor(values, &node.input[1])?;

    // Handle legacy ONNX broadcast attribute: when broadcast=1 and axis is set,
    // the second operand's shape is aligned starting at the given axis.
    // e.g. a=[1,8,24,24], b=[8], axis=1 => reshape b to [1,8,1,1]
    let legacy_broadcast = get_attr_int(node, "broadcast").unwrap_or(0);
    if legacy_broadcast == 1 && b.dims.len() < a.dims.len() {
        let axis = get_attr_int(node, "axis").unwrap_or(0) as usize;
        let mut new_dims = vec![1usize; a.dims.len()];
        for (i, &d) in b.dims.iter().enumerate() {
            new_dims[axis + i] = d;
        }
        b.dims = new_dims;
    }

    let out_shape = broadcast_shape(&a.dims, &b.dims);
    let numel: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; numel];

    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];

    for val in &mut data {
        let ai = broadcast_index(&index, &a.dims, &out_shape);
        let bi = broadcast_index(&index, &b.dims, &out_shape);
        *val = op(a.data[ai], b.data[bi]);

        // Increment index
        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(out_shape, data));
    Ok(())
}

pub fn exec_div(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a / b)
}

pub fn exec_sub(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a - b)
}

pub fn exec_mul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a * b)
}

pub fn exec_add(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    exec_binary_op(node, values, |a, b| a + b)
}
