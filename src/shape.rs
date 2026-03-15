use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_attr_int;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::onnx::NodeProto;

pub fn exec_constant(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| InferenceError::InvalidModel("Constant has no value".into()))?;

    let tensor_proto = attr
        .t
        .as_ref()
        .ok_or_else(|| InferenceError::InvalidModel("Constant value is not a tensor".into()))?;

    let tensor = Tensor::from_proto(tensor_proto)?;
    values.insert(node.output[0].clone(), tensor);
    Ok(())
}

pub fn exec_reshape(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    // Shape can come from attribute or second input
    let new_shape: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let shape_tensor = get_tensor(values, &node.input[1])?;
        shape_tensor.data.iter().map(|&v| v as i64).collect()
    } else {
        // Check for shape attribute (older ONNX format)
        get_attr_ints(node, "shape").ok_or_else(|| {
            InferenceError::InvalidModel("Reshape: no shape input or attribute".into())
        })?
    };

    let total = input.numel();
    let mut dims: Vec<usize> = Vec::new();
    let mut infer_idx: Option<usize> = None;

    for (i, &s) in new_shape.iter().enumerate() {
        if s == -1 {
            infer_idx = Some(i);
            dims.push(0); // placeholder
        } else if s == 0 {
            dims.push(input.dims[i]);
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
        dims[idx] = total / known;
    }

    values.insert(node.output[0].clone(), Tensor::new(dims, input.data));
    Ok(())
}

pub fn exec_flatten(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;

    let outer: usize = input.dims[..axis].iter().product();
    let inner: usize = input.dims[axis..].iter().product();

    values.insert(
        node.output[0].clone(),
        Tensor::new(vec![outer, inner], input.data),
    );
    Ok(())
}

pub fn exec_shape(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let rank = input.dims.len();
    let data: Vec<f32> = input.dims.iter().map(|&d| d as f32).collect();
    values.insert(node.output[0].clone(), Tensor::new(vec![rank], data));
    Ok(())
}

pub fn exec_gather(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let indices = get_tensor(values, &node.input[1])?;
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    // For scalar index gathering from a 1-D tensor (common case: Shape -> Gather)
    if indices.dims.is_empty()
        || (indices.dims.len() == 1 && indices.dims[0] == 1)
        || indices.numel() == 1
    {
        let idx = indices.data[0] as i64;
        let idx = if idx < 0 {
            input.dims[axis] as i64 + idx
        } else {
            idx
        } as usize;

        if input.dims.len() == 1 {
            // Scalar output
            values.insert(
                node.output[0].clone(),
                Tensor::new(vec![], vec![input.data[idx]]),
            );
        } else {
            // Slice along axis
            let outer: usize = input.dims[..axis].iter().product();
            let inner: usize = input.dims[axis + 1..].iter().product();
            let axis_size = input.dims[axis];
            let mut data = Vec::with_capacity(outer * inner);
            for o in 0..outer {
                let base = o * axis_size * inner + idx * inner;
                data.extend_from_slice(&input.data[base..base + inner]);
            }
            let mut out_dims: Vec<usize> = input.dims[..axis].to_vec();
            out_dims.extend_from_slice(&input.dims[axis + 1..]);
            if out_dims.is_empty() {
                out_dims = vec![];
            }
            values.insert(node.output[0].clone(), Tensor::new(out_dims, data));
        }
    } else {
        return Err(InferenceError::UnsupportedOperator(
            "Gather with non-scalar indices".into(),
        ));
    }
    Ok(())
}

pub fn exec_unsqueeze(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    // Opset 13+: axes from second input; earlier: from attribute
    let axes: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_tensor = get_tensor(values, &node.input[1])?;
        axes_tensor.data.iter().map(|&v| v as i64).collect()
    } else {
        get_attr_ints(node, "axes").unwrap_or_default()
    };

    let out_rank = input.dims.len() + axes.len();
    let mut out_dims = input.dims.clone();
    let mut sorted_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (out_rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    sorted_axes.sort();
    for &ax in &sorted_axes {
        out_dims.insert(ax, 1);
    }

    values.insert(node.output[0].clone(), Tensor::new(out_dims, input.data));
    Ok(())
}

pub fn exec_concat(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let axis = get_attr_int(node, "axis").unwrap_or(0);

    let tensors: Vec<Tensor> = node
        .input
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| get_tensor(values, name))
        .collect::<Result<Vec<_>>>()?;

    if tensors.is_empty() {
        return Err(InferenceError::InvalidModel("Concat with no inputs".into()));
    }

    let rank = tensors[0].dims.len();
    let axis = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let mut out_dims = tensors[0].dims.clone();
    out_dims[axis] = tensors.iter().map(|t| t.dims[axis]).sum();

    let outer: usize = out_dims[..axis].iter().product();
    let inner: usize = out_dims[axis + 1..].iter().product();
    let total = out_dims.iter().product::<usize>();
    let mut data = vec![0.0f32; total];

    let mut axis_offset = 0;
    for t in &tensors {
        let t_axis = t.dims[axis];
        for o in 0..outer {
            for a in 0..t_axis {
                let src_base = (o * t_axis + a) * inner;
                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                data[dst_base..dst_base + inner]
                    .copy_from_slice(&t.data[src_base..src_base + inner]);
            }
        }
        axis_offset += t_axis;
    }

    values.insert(node.output[0].clone(), Tensor::new(out_dims, data));
    Ok(())
}
