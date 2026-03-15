use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_attr_float;
use crate::get_attr_int;
use crate::get_tensor;
use crate::onnx::NodeProto;

pub fn exec_relu(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let data: Vec<f32> = input.floats().iter().map(|&v| v.max(0.0)).collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_leaky_relu(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let alpha = get_attr_float(node, "alpha").unwrap_or(0.01);
    let data: Vec<f32> = input
        .floats()
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_clip(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    // Opset 11+: min/max are optional inputs (inputs[1], inputs[2])
    // Opset 6: min/max are attributes
    let min_val = if node.input.len() > 1 && !node.input[1].is_empty() {
        get_tensor(values, &node.input[1])?.floats()[0]
    } else {
        get_attr_float(node, "min").unwrap_or(f32::NEG_INFINITY)
    };
    let max_val = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?.floats()[0]
    } else {
        get_attr_float(node, "max").unwrap_or(f32::INFINITY)
    };

    let data: Vec<f32> = input
        .floats()
        .iter()
        .map(|&v| v.clamp(min_val, max_val))
        .collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_batch_normalization(
    node: &NodeProto,
    values: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let scale = get_tensor(values, &node.input[1])?;
    let bias = get_tensor(values, &node.input[2])?;
    let mean = get_tensor(values, &node.input[3])?;
    let var = get_tensor(values, &node.input[4])?;
    let epsilon = get_attr_float(node, "epsilon").unwrap_or(1e-5);

    // input: [N, C, ...spatial...]
    let n = input.dims[0];
    let c = input.dims[1];
    let spatial: usize = input.dims[2..].iter().product();

    let mut data = input.floats().to_vec();
    let scale_f = scale.floats();
    let bias_f = bias.floats();
    let mean_f = mean.floats();
    let var_f = var.floats();
    for batch in 0..n {
        for ch in 0..c {
            let s = scale_f[ch];
            let b = bias_f[ch];
            let m = mean_f[ch];
            let v = var_f[ch];
            let inv_std = 1.0 / (v + epsilon).sqrt();
            let base = (batch * c + ch) * spatial;
            for i in 0..spatial {
                data[base + i] = (data[base + i] - m) * inv_std * s + b;
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_sigmoid(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let data: Vec<f32> = input
        .floats()
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_exp(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let data: Vec<f32> = input.floats().iter().map(|&v| v.exp()).collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_ceil(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let data: Vec<f32> = input.floats().iter().map(|&v| v.ceil()).collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_round(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    // ONNX Round uses "round half to even" (banker's rounding)
    let data: Vec<f32> = input
        .floats()
        .iter()
        .map(|&v| {
            let rounded = v.round();
            // Handle .5 case: round to even
            if (v - v.floor() - 0.5).abs() < f32::EPSILON {
                if rounded as i64 % 2 != 0 {
                    rounded - 1.0
                } else {
                    rounded
                }
            } else {
                rounded
            }
        })
        .collect();
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_softmax(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(-1);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let outer: usize = input.dims[..axis].iter().product();
    let dim = input.dims[axis];
    let inner: usize = input.dims[axis + 1..].iter().product();

    let mut data = input.floats().to_vec();

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                max_val = max_val.max(data[idx]);
            }
            let mut sum = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                data[idx] = (data[idx] - max_val).exp();
                sum += data[idx];
            }
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                data[idx] /= sum;
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}
