use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::DType;
use crate::execute_node;
use crate::get_attr_int;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::onnx::GraphProto;
use crate::onnx::NodeProto;

pub fn exec_constant(
    node: &NodeProto,
    _values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
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
    *output = tensor;
    Ok(())
}

pub fn exec_reshape(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    let shape_from_attr;
    let new_shape: &[i64] = if node.input.len() > 1 && !node.input[1].is_empty() {
        let shape_tensor = get_tensor(values, &node.input[1])?;
        shape_tensor.ints()
    } else {
        shape_from_attr = get_attr_ints(node, "shape").ok_or_else(|| {
            InferenceError::InvalidModel("Reshape: no shape input or attribute".into())
        })?;
        &shape_from_attr
    };

    let total = input.numel();
    let mut dims: Vec<usize> = Vec::new();
    let mut infer_idx: Option<usize> = None;

    for (i, &s) in new_shape.iter().enumerate() {
        if s == -1 {
            infer_idx = Some(i);
            dims.push(0);
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

    output.copy_from(input);
    output.dims = dims;
    Ok(())
}

pub fn exec_flatten(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;

    let outer: usize = input.dims[..axis].iter().product();
    let inner: usize = input.dims[axis..].iter().product();

    output.copy_from(input);
    output.dims = vec![outer, inner];
    Ok(())
}

pub fn exec_shape(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let rank = input.dims.len();
    let buf = output.as_mut_i64(rank);
    for (o, &d) in buf.iter_mut().zip(input.dims.iter()) {
        *o = d as i64;
    }
    output.dims = vec![rank];
    Ok(())
}

pub fn exec_gather(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let indices = get_tensor(values, &node.input[1])?;
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    if indices.dims.is_empty()
        || (indices.dims.len() == 1 && indices.dims[0] == 1)
        || indices.numel() == 1
    {
        let idx_i64 = indices.i64_at(0);
        let idx = if idx_i64 < 0 {
            (input.dims[axis] as i64 + idx_i64) as usize
        } else {
            idx_i64 as usize
        };

        if input.dims.len() == 1 {
            match input.dtype {
                DType::Float => {
                    let buf = output.as_mut_f32(1);
                    buf[0] = input.floats()[idx];
                }
                DType::Int64 => {
                    let buf = output.as_mut_i64(1);
                    buf[0] = input.ints()[idx];
                }
            }
            output.dims = vec![];
        } else {
            let outer: usize = input.dims[..axis].iter().product();
            let inner: usize = input.dims[axis + 1..].iter().product();
            let axis_size = input.dims[axis];
            let mut out_dims: Vec<usize> = input.dims[..axis].to_vec();
            out_dims.extend_from_slice(&input.dims[axis + 1..]);

            match input.dtype {
                DType::Float => {
                    let d = input.floats();
                    let buf = output.as_mut_f32(outer * inner);
                    for o in 0..outer {
                        let base = o * axis_size * inner + idx * inner;
                        let dst = o * inner;
                        buf[dst..dst + inner].copy_from_slice(&d[base..base + inner]);
                    }
                }
                DType::Int64 => {
                    let d = input.ints();
                    let buf = output.as_mut_i64(outer * inner);
                    for o in 0..outer {
                        let base = o * axis_size * inner + idx * inner;
                        let dst = o * inner;
                        buf[dst..dst + inner].copy_from_slice(&d[base..base + inner]);
                    }
                }
            }
            output.dims = out_dims;
        }
    } else {
        return Err(InferenceError::UnsupportedOperator(
            "Gather with non-scalar indices".into(),
        ));
    }
    Ok(())
}

pub fn exec_unsqueeze(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    let axes_from_attr;
    let axes: &[i64] = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_tensor = get_tensor(values, &node.input[1])?;
        axes_tensor.ints()
    } else {
        axes_from_attr = get_attr_ints(node, "axes").unwrap_or_default();
        &axes_from_attr
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

    output.copy_from(input);
    output.dims = out_dims;
    Ok(())
}

pub fn exec_concat(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let axis = get_attr_int(node, "axis").unwrap_or(0);

    let tensors: Vec<&Tensor> = node
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

    let is_int = tensors[0].dtype == DType::Int64;
    if is_int {
        let total = out_dims.iter().product::<usize>();
        let buf = output.as_mut_i64(total);
        let mut axis_offset = 0;
        for t in &tensors {
            let t_data = t.ints();
            let t_axis = t.dims[axis];
            for o in 0..outer {
                for a in 0..t_axis {
                    let src_base = (o * t_axis + a) * inner;
                    let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                    buf[dst_base..dst_base + inner]
                        .copy_from_slice(&t_data[src_base..src_base + inner]);
                }
            }
            axis_offset += t_axis;
        }
    } else {
        let total = out_dims.iter().product::<usize>();
        let buf = output.as_mut_f32(total);
        let mut axis_offset = 0;
        for t in &tensors {
            let t_data = t.floats();
            let t_axis = t.dims[axis];
            for o in 0..outer {
                for a in 0..t_axis {
                    let src_base = (o * t_axis + a) * inner;
                    let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                    buf[dst_base..dst_base + inner]
                        .copy_from_slice(&t_data[src_base..src_base + inner]);
                }
            }
            axis_offset += t_axis;
        }
    }
    output.dims = out_dims;
    Ok(())
}

pub fn exec_identity(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    output.copy_from(input);
    Ok(())
}

pub fn exec_cast(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let to = get_attr_int(node, "to").unwrap_or(1);
    let numel = input.numel();
    match to {
        6 | 7 => {
            let buf = output.as_mut_i64(numel);
            match input.dtype {
                DType::Int64 => buf.copy_from_slice(input.ints()),
                DType::Float => {
                    for (o, &v) in buf.iter_mut().zip(input.floats().iter()) {
                        *o = v as i64;
                    }
                }
            }
            output.dims.clone_from(&input.dims);
        }
        _ => {
            let buf = output.as_mut_f32(numel);
            match input.dtype {
                DType::Float => buf.copy_from_slice(input.floats()),
                DType::Int64 => {
                    for (o, &v) in buf.iter_mut().zip(input.ints().iter()) {
                        *o = v as f32;
                    }
                }
            }
            output.dims.clone_from(&input.dims);
        }
    }
    Ok(())
}

pub fn exec_transpose(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let rank = input.dims.len();

    let perm: Vec<usize> = get_attr_ints(node, "perm")
        .map(|p| p.iter().map(|&v| v as usize).collect())
        .unwrap_or_else(|| (0..rank).rev().collect());

    let mut out_dims = vec![0usize; rank];
    for i in 0..rank {
        out_dims[i] = input.dims[perm[i]];
    }

    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
    }

    let numel = input.numel();
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    match input.dtype {
        DType::Float => {
            let in_data = input.floats();
            let buf = output.as_mut_f32(numel);
            for out_flat in 0..numel {
                let mut remaining = out_flat;
                let mut in_flat = 0;
                for i in 0..rank {
                    let coord = remaining / out_strides[i];
                    remaining %= out_strides[i];
                    in_flat += coord * in_strides[perm[i]];
                }
                buf[out_flat] = in_data[in_flat];
            }
        }
        DType::Int64 => {
            let in_data = input.ints();
            let buf = output.as_mut_i64(numel);
            for out_flat in 0..numel {
                let mut remaining = out_flat;
                let mut in_flat = 0;
                for i in 0..rank {
                    let coord = remaining / out_strides[i];
                    remaining %= out_strides[i];
                    in_flat += coord * in_strides[perm[i]];
                }
                buf[out_flat] = in_data[in_flat];
            }
        }
    }
    output.dims = out_dims;
    Ok(())
}

pub fn exec_squeeze(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    let axes_from_attr;
    let axes: &[i64] = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_tensor = get_tensor(values, &node.input[1])?;
        axes_tensor.ints()
    } else {
        axes_from_attr = get_attr_ints(node, "axes").unwrap_or_default();
        &axes_from_attr
    };

    let rank = input.dims.len() as i64;
    let axes_set: std::collections::HashSet<usize> = if axes.is_empty() {
        input
            .dims
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 1)
            .map(|(i, _)| i)
            .collect()
    } else {
        axes.iter()
            .map(|&a| {
                if a < 0 {
                    (rank + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    };

    let out_dims: Vec<usize> = input
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes_set.contains(i))
        .map(|(_, &d)| d)
        .collect();

    output.copy_from(input);
    output.dims = out_dims;
    Ok(())
}

pub fn exec_slice(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let starts_t = get_tensor(values, &node.input[1])?;
    let ends_t = get_tensor(values, &node.input[2])?;
    let axes_t = if node.input.len() > 3 && !node.input[3].is_empty() {
        Some(get_tensor(values, &node.input[3])?)
    } else {
        None
    };
    let steps_t = if node.input.len() > 4 && !node.input[4].is_empty() {
        Some(get_tensor(values, &node.input[4])?)
    } else {
        None
    };

    let rank = input.dims.len();
    let mut starts = vec![0i64; rank];
    let mut ends: Vec<i64> = input.dims.iter().map(|&d| d as i64).collect();
    let mut steps = vec![1i64; rank];

    let starts_ints = starts_t.ints();
    let ends_ints = ends_t.ints();

    let axes: Vec<usize> = if let Some(ref at) = axes_t {
        at.ints()
            .iter()
            .map(|&a| {
                if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else {
        (0..starts_ints.len()).collect()
    };

    for (i, &ax) in axes.iter().enumerate() {
        starts[ax] = starts_ints[i];
        ends[ax] = ends_ints[i];
        if let Some(ref st) = steps_t {
            steps[ax] = st.ints()[i];
        }
    }

    // Clamp starts/ends per ONNX Slice spec
    for ax in 0..rank {
        let dim = input.dims[ax] as i64;
        if steps[ax] > 0 {
            if starts[ax] < 0 {
                starts[ax] += dim;
            }
            if ends[ax] < 0 {
                ends[ax] += dim;
            }
            starts[ax] = starts[ax].clamp(0, dim);
            ends[ax] = ends[ax].clamp(0, dim);
        } else {
            if starts[ax] < 0 {
                starts[ax] += dim;
            }
            if ends[ax] < -dim {
                ends[ax] = -1;
            } else if ends[ax] < 0 {
                ends[ax] += dim;
            }
            starts[ax] = starts[ax].clamp(-1, dim - 1);
            ends[ax] = ends[ax].clamp(-1, dim - 1);
        }
    }

    // Compute output dims
    let mut out_dims = vec![0usize; rank];
    for ax in 0..rank {
        if steps[ax] > 0 {
            if ends[ax] <= starts[ax] {
                out_dims[ax] = 0;
            } else {
                out_dims[ax] = ((ends[ax] - starts[ax] - 1) / steps[ax] + 1) as usize;
            }
        } else if starts[ax] <= ends[ax] {
            out_dims[ax] = 0;
        } else {
            out_dims[ax] = ((starts[ax] - ends[ax] - 1) / (-steps[ax]) + 1) as usize;
        }
    }

    let numel: usize = out_dims.iter().product();

    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
    }

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    match input.dtype {
        DType::Float => {
            let in_data = input.floats();
            let buf = output.as_mut_f32(numel);
            for out_flat in 0..numel {
                let mut remaining = out_flat;
                let mut in_flat = 0;
                for ax in 0..rank {
                    let coord = remaining / out_strides[ax];
                    remaining %= out_strides[ax];
                    let in_coord = starts[ax] + coord as i64 * steps[ax];
                    in_flat += in_coord as usize * in_strides[ax];
                }
                buf[out_flat] = in_data[in_flat];
            }
        }
        DType::Int64 => {
            let in_data = input.ints();
            let buf = output.as_mut_i64(numel);
            for out_flat in 0..numel {
                let mut remaining = out_flat;
                let mut in_flat = 0;
                for ax in 0..rank {
                    let coord = remaining / out_strides[ax];
                    remaining %= out_strides[ax];
                    let in_coord = starts[ax] + coord as i64 * steps[ax];
                    in_flat += in_coord as usize * in_strides[ax];
                }
                buf[out_flat] = in_data[in_flat];
            }
        }
    }
    output.dims = out_dims;
    Ok(())
}

pub fn exec_tile(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let repeats_t = get_tensor(values, &node.input[1])?;
    let repeats: Vec<usize> = repeats_t.ints().iter().map(|&v| v as usize).collect();

    let rank = input.dims.len();
    let mut out_dims = vec![0usize; rank];
    for i in 0..rank {
        out_dims[i] = input.dims[i] * repeats[i];
    }

    let numel: usize = out_dims.iter().product();

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
    }

    match input.dtype {
        DType::Float => {
            let in_data = input.floats();
            let buf = output.as_mut_f32(numel);
            for out_flat in 0..numel {
                let mut remaining = out_flat;
                let mut in_flat = 0;
                for ax in 0..rank {
                    let coord = remaining / out_strides[ax];
                    remaining %= out_strides[ax];
                    in_flat += (coord % input.dims[ax]) * in_strides[ax];
                }
                buf[out_flat] = in_data[in_flat];
            }
        }
        DType::Int64 => {
            let in_data = input.ints();
            let buf = output.as_mut_i64(numel);
            for out_flat in 0..numel {
                let mut remaining = out_flat;
                let mut in_flat = 0;
                for ax in 0..rank {
                    let coord = remaining / out_strides[ax];
                    remaining %= out_strides[ax];
                    in_flat += (coord % input.dims[ax]) * in_strides[ax];
                }
                buf[out_flat] = in_data[in_flat];
            }
        }
    }
    output.dims = out_dims;
    Ok(())
}

pub fn exec_resize(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    let rank = input.dims.len();

    let out_dims: Vec<usize> = if node.input.len() > 3 && !node.input[3].is_empty() {
        let sizes = get_tensor(values, &node.input[3])?;
        sizes.ints().iter().map(|&v| v as usize).collect()
    } else if node.input.len() > 2 && !node.input[2].is_empty() {
        let scales = get_tensor(values, &node.input[2])?;
        let scales_f = scales.floats();
        input
            .dims
            .iter()
            .zip(scales_f.iter())
            .map(|(&d, &s)| (d as f32 * s) as usize)
            .collect()
    } else {
        return Err(InferenceError::InvalidModel(
            "Resize: no scales or sizes".into(),
        ));
    };

    let numel: usize = out_dims.iter().product();

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
    }

    let input_f = input.floats();
    let buf = output.as_mut_f32(numel);
    for out_flat in 0..numel {
        let mut remaining = out_flat;
        let mut in_flat = 0;
        for ax in 0..rank {
            let out_coord = remaining / out_strides[ax];
            remaining %= out_strides[ax];
            let scale = out_dims[ax] as f32 / input.dims[ax] as f32;
            let in_coord = ((out_coord as f32 + 0.5) / scale - 0.5)
                .round()
                .max(0.0)
                .min((input.dims[ax] - 1) as f32) as usize;
            in_flat += in_coord * in_strides[ax];
        }
        buf[out_flat] = input_f[in_flat];
    }

    output.dims = out_dims;
    Ok(())
}

pub fn exec_reduce_min(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) != 0;

    let axes_from_attr;
    let axes: &[i64] = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_t = get_tensor(values, &node.input[1])?;
        axes_t.ints()
    } else {
        axes_from_attr =
            get_attr_ints(node, "axes").unwrap_or_else(|| (0..input.dims.len() as i64).collect());
        &axes_from_attr
    };

    let rank = input.dims.len() as i64;
    let axes_set: std::collections::HashSet<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (rank + a) as usize
            } else {
                a as usize
            }
        })
        .collect();

    let mut out_dims = Vec::new();
    for (i, &d) in input.dims.iter().enumerate() {
        if axes_set.contains(&i) {
            if keepdims {
                out_dims.push(1);
            }
        } else {
            out_dims.push(d);
        }
    }
    if out_dims.is_empty() {
        out_dims.push(1);
    }

    let out_numel: usize = out_dims.iter().product();
    let buf = output.as_mut_f32(out_numel);
    buf.fill(f32::INFINITY);

    let in_rank = input.dims.len();
    let mut in_strides = vec![1usize; in_rank];
    for i in (0..in_rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
    }

    let out_rank = out_dims.len();
    let mut out_strides = vec![1usize; out_rank];
    if out_rank > 1 {
        for i in (0..out_rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }
    }

    let input_f = input.floats();
    for in_flat in 0..input.numel() {
        let mut remaining = in_flat;
        let mut out_flat = 0;
        let mut out_idx = 0;
        for ax in 0..in_rank {
            let coord = remaining / in_strides[ax];
            remaining %= in_strides[ax];
            if !axes_set.contains(&ax) {
                out_flat += coord * out_strides[out_idx];
                out_idx += 1;
            } else if keepdims {
                out_idx += 1;
            }
        }
        buf[out_flat] = buf[out_flat].min(input_f[in_flat]);
    }

    output.dims = out_dims;
    Ok(())
}

pub fn exec_loop(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let body: &GraphProto = node
        .attribute
        .iter()
        .find(|a| a.name == "body")
        .and_then(|a| a.g.as_ref())
        .ok_or_else(|| InferenceError::InvalidModel("Loop: no body graph".into()))?;

    let trip_tensor = get_tensor(values, &node.input[0])?;
    let trip_count = trip_tensor.i64_at(0) as usize;

    let num_carried = node.input.len() - 2;
    let mut carried: Vec<Tensor> = (0..num_carried)
        .map(|i| get_tensor(values, &node.input[i + 2]).cloned())
        .collect::<Result<Vec<_>>>()?;

    let num_scan = body.output.len() - 1 - num_carried;
    let mut scan_outputs: Vec<Vec<Tensor>> = vec![vec![]; num_scan];

    for i in 0..trip_count {
        let mut body_values: HashMap<String, Tensor> = HashMap::new();

        for (k, v) in values.iter() {
            body_values.insert(k.clone(), v.clone());
        }

        body_values.insert(
            body.input[0].name.clone(),
            Tensor::new_i64(vec![], vec![i as i64]),
        );
        body_values.insert(body.input[1].name.clone(), Tensor::new(vec![], vec![1.0]));
        for (j, c) in carried.iter().enumerate() {
            body_values.insert(body.input[j + 2].name.clone(), c.clone());
        }

        for init in &body.initializer {
            if !init.name.is_empty() {
                body_values.insert(init.name.clone(), Tensor::from_proto(init)?);
            }
        }

        for body_node in &body.node {
            execute_node(body_node, &mut body_values)?;
        }

        for j in 0..num_carried {
            carried[j] = body_values
                .get(&body.output[j + 1].name)
                .cloned()
                .unwrap_or_else(|| Tensor::new(vec![], vec![]));
        }

        for j in 0..num_scan {
            let scan_val = body_values
                .get(&body.output[1 + num_carried + j].name)
                .cloned()
                .unwrap_or_else(|| Tensor::new(vec![], vec![]));
            scan_outputs[j].push(scan_val);
        }
    }

    let mut out_idx = 0;
    for j in 0..num_carried {
        if out_idx < node.output.len() && !node.output[out_idx].is_empty() {
            values.insert(node.output[out_idx].clone(), carried[j].clone());
        }
        out_idx += 1;
    }
    for j in 0..num_scan {
        if out_idx < node.output.len() && !node.output[out_idx].is_empty() {
            let scans = &scan_outputs[j];
            if scans.is_empty() {
                values.insert(node.output[out_idx].clone(), Tensor::new(vec![0], vec![]));
            } else {
                let inner_dims = &scans[0].dims;
                let mut out_dims = vec![scans.len()];
                out_dims.extend_from_slice(inner_dims);
                match scans[0].dtype {
                    DType::Float => {
                        let data: Vec<f32> = scans
                            .iter()
                            .flat_map(|t| {
                                (0..t.numel()).map(|i| t.f32_at(i))
                            })
                            .collect();
                        values.insert(node.output[out_idx].clone(), Tensor::new(out_dims, data));
                    }
                    DType::Int64 => {
                        let data: Vec<i64> = scans
                            .iter()
                            .flat_map(|t| {
                                (0..t.numel()).map(|i| t.i64_at(i))
                            })
                            .collect();
                        values.insert(
                            node.output[out_idx].clone(),
                            Tensor::new_i64(out_dims, data),
                        );
                    }
                }
            }
        }
        out_idx += 1;
    }

    Ok(())
}

pub fn exec_nms(
    node: &NodeProto,
    values: &HashMap<String, Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    let boxes = get_tensor(values, &node.input[0])?;
    let scores = get_tensor(values, &node.input[1])?;
    let max_output = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?.i64_at(0) as usize
    } else {
        0
    };
    let iou_threshold = if node.input.len() > 3 && !node.input[3].is_empty() {
        get_tensor(values, &node.input[3])?.floats()[0]
    } else {
        0.0
    };
    let score_threshold = if node.input.len() > 4 && !node.input[4].is_empty() {
        get_tensor(values, &node.input[4])?.floats()[0]
    } else {
        f32::NEG_INFINITY
    };

    let batches = boxes.dims[0];
    let num_boxes = boxes.dims[1];
    let num_classes = scores.dims[1];

    let boxes_f = boxes.floats();
    let scores_f = scores.floats();
    let mut selected: Vec<[i64; 3]> = Vec::new();

    for batch in 0..batches {
        for class in 0..num_classes {
            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for b in 0..num_boxes {
                let score = scores_f[(batch * num_classes + class) * num_boxes + b];
                if score > score_threshold {
                    candidates.push((b, score));
                }
            }
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut kept: Vec<usize> = Vec::new();
            for &(box_idx, _) in &candidates {
                if max_output > 0 && kept.len() >= max_output {
                    break;
                }
                let base_i = (batch * num_boxes + box_idx) * 4;
                let bi = &boxes_f[base_i..base_i + 4];

                let mut suppress = false;
                for &k in &kept {
                    let base_k = (batch * num_boxes + k) * 4;
                    let bk = &boxes_f[base_k..base_k + 4];
                    if iou(bi, bk) > iou_threshold {
                        suppress = true;
                        break;
                    }
                }
                if !suppress {
                    kept.push(box_idx);
                    selected.push([batch as i64, class as i64, box_idx as i64]);
                }
            }
        }
    }

    let num_selected = selected.len();
    let buf = output.as_mut_i64(num_selected * 3);
    for (i, s) in selected.iter().enumerate() {
        buf[i * 3] = s[0];
        buf[i * 3 + 1] = s[1];
        buf[i * 3 + 2] = s[2];
    }
    output.dims = vec![num_selected, 3];
    Ok(())
}

fn iou(a: &[f32], b: &[f32]) -> f32 {
    let y1 = a[0].max(b[0]);
    let x1 = a[1].max(b[1]);
    let y2 = a[2].min(b[2]);
    let x2 = a[3].min(b[3]);
    let inter = (y2 - y1).max(0.0) * (x2 - x1).max(0.0);
    let area_a = (a[2] - a[0]).abs() * (a[3] - a[1]).abs();
    let area_b = (b[2] - b[0]).abs() * (b[3] - b[1]).abs();
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}
