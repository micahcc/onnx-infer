use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape;
use crate::conv::exec_conv;
use crate::get_attr_int;
use crate::get_tensor;
use crate::matmul::exec_matmul;
use crate::onnx::NodeProto;
use crate::pool::exec_global_avg_pool;

/// Dequantize: float = (quantized - zero_point) * scale
fn dequantize(data: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter().map(|&v| (v - zero_point) * scale).collect()
}

/// Quantize: quantized = clamp(round(float / scale) + zero_point, 0, 255)
fn quantize_u8(data: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter()
        .map(|&v| (v / scale + zero_point).round().clamp(0.0, 255.0))
        .collect()
}

pub fn exec_quantize_linear(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let scale = get_tensor(values, &node.input[1])?;
    let zero_point = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?.data[0]
    } else {
        0.0
    };
    let data = quantize_u8(&input.data, scale.data[0], zero_point);
    values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    Ok(())
}

pub fn exec_dequantize_linear(
    node: &NodeProto,
    values: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    let scale = get_tensor(values, &node.input[1])?;
    let zero_point = if node.input.len() > 2 && !node.input[2].is_empty() {
        get_tensor(values, &node.input[2])?
    } else {
        Tensor::new(vec![], vec![0.0])
    };

    let axis = get_attr_int(node, "axis").unwrap_or(1);
    let rank = input.dims.len() as i64;
    let axis = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    if scale.data.len() == 1 {
        // Scalar dequant
        let data = dequantize(&input.data, scale.data[0], zero_point.data[0]);
        values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    } else {
        // Per-channel dequant along axis
        let outer: usize = input.dims[..axis].iter().product();
        let ch = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();
        let mut data = vec![0.0f32; input.data.len()];
        for o in 0..outer {
            for c in 0..ch {
                let s = scale.data[c];
                let zp = zero_point.data[c];
                let base = (o * ch + c) * inner;
                for i in 0..inner {
                    data[base + i] = (input.data[base + i] - zp) * s;
                }
            }
        }
        values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    }
    Ok(())
}

/// QLinearConv: quantized conv with dequantize-compute-requantize pattern.
/// Inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [bias]
/// Weight scale/zp can be per-channel (one per output channel).
pub fn exec_qlinear_conv(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.data[0];
    let x_zp = get_tensor(values, &node.input[2])?.data[0];
    let w_quant = get_tensor(values, &node.input[3])?;
    let w_scale_t = get_tensor(values, &node.input[4])?;
    let w_zp_t = get_tensor(values, &node.input[5])?;
    let y_scale = get_tensor(values, &node.input[6])?.data[0];
    let y_zp = get_tensor(values, &node.input[7])?.data[0];
    let bias = if node.input.len() > 8 && !node.input[8].is_empty() {
        Some(get_tensor(values, &node.input[8])?)
    } else {
        None
    };

    // Dequantize x (always scalar scale/zp)
    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(&x_quant.data, x_scale, x_zp),
    );

    // Dequantize w: per-channel along axis 0 (output channels)
    let c_out = w_quant.dims[0];
    let per_channel = w_scale_t.data.len() > 1;
    let elems_per_oc = w_quant.data.len() / c_out;
    let mut w_float_data = vec![0.0f32; w_quant.data.len()];
    for oc in 0..c_out {
        let scale = if per_channel {
            w_scale_t.data[oc]
        } else {
            w_scale_t.data[0]
        };
        let zp = if per_channel {
            w_zp_t.data[oc]
        } else {
            w_zp_t.data[0]
        };
        let base = oc * elems_per_oc;
        for i in 0..elems_per_oc {
            w_float_data[base + i] = (w_quant.data[base + i] - zp) * scale;
        }
    }
    let w_float = Tensor::new(w_quant.dims.clone(), w_float_data);

    // Build a synthetic Conv node reusing attributes from this node
    let mut conv_node = node.clone();
    conv_node.op_type = "Conv".to_string();
    conv_node.input = vec!["__qconv_x__".to_string(), "__qconv_w__".to_string()];
    if bias.is_some() {
        conv_node.input.push("__qconv_b__".to_string());
    }
    conv_node.output = vec!["__qconv_y__".to_string()];

    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qconv_x__".to_string(), x_float);
    tmp_values.insert("__qconv_w__".to_string(), w_float);
    if let Some(b) = bias {
        // Bias is int32. bias_float[oc] = bias_int32[oc] * x_scale * w_scale[oc]
        let bias_float: Vec<f32> = b
            .data
            .iter()
            .enumerate()
            .map(|(oc, &val)| {
                let ws = if per_channel {
                    w_scale_t.data[oc]
                } else {
                    w_scale_t.data[0]
                };
                val * x_scale * ws
            })
            .collect();
        tmp_values.insert("__qconv_b__".to_string(), Tensor::new(b.dims, bias_float));
    }

    exec_conv(&conv_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qconv_y__").unwrap();

    // Requantize output
    let y_quant = quantize_u8(&y_float.data, y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

/// QLinearAdd: inputs x, x_scale, x_zp, y, y_scale, y_zp, z_scale, z_zp
pub fn exec_qlinear_add(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.data[0];
    let x_zp = get_tensor(values, &node.input[2])?.data[0];
    let y_quant = get_tensor(values, &node.input[3])?;
    let y_scale = get_tensor(values, &node.input[4])?.data[0];
    let y_zp = get_tensor(values, &node.input[5])?.data[0];
    let z_scale = get_tensor(values, &node.input[6])?.data[0];
    let z_zp = get_tensor(values, &node.input[7])?.data[0];

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(&x_quant.data, x_scale, x_zp),
    );
    let y_float = Tensor::new(
        y_quant.dims.clone(),
        dequantize(&y_quant.data, y_scale, y_zp),
    );

    // Broadcast add
    let out_shape = broadcast_shape(&x_float.dims, &y_float.dims);
    let numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];
    let mut z_float = vec![0.0f32; numel];

    for val in &mut z_float {
        let ai = broadcast_index(&index, &x_float.dims, &out_shape);
        let bi = broadcast_index(&index, &y_float.dims, &out_shape);
        *val = x_float.data[ai] + y_float.data[bi];
        for d in (0..ndim).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }

    let z_quant = quantize_u8(&z_float, z_scale, z_zp);
    values.insert(node.output[0].clone(), Tensor::new(out_shape, z_quant));
    Ok(())
}

/// QLinearMatMul: inputs a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
/// Weight (b) scale/zp can be per-channel (one per last-axis column).
pub fn exec_qlinear_matmul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a_quant = get_tensor(values, &node.input[0])?;
    let a_scale = get_tensor(values, &node.input[1])?.data[0];
    let a_zp = get_tensor(values, &node.input[2])?.data[0];
    let b_quant = get_tensor(values, &node.input[3])?;
    let b_scale_t = get_tensor(values, &node.input[4])?;
    let b_zp_t = get_tensor(values, &node.input[5])?;
    let y_scale = get_tensor(values, &node.input[6])?.data[0];
    let y_zp = get_tensor(values, &node.input[7])?.data[0];

    let a_float = Tensor::new(
        a_quant.dims.clone(),
        dequantize(&a_quant.data, a_scale, a_zp),
    );

    // Dequantize b: per-channel along last axis if scale is a vector
    let b_float = if b_scale_t.data.len() > 1 {
        // Per-channel: b shape is [..., K, N], scale shape is [N]
        let n = *b_quant.dims.last().unwrap();
        let k = b_quant.data.len() / n;
        let mut data = vec![0.0f32; b_quant.data.len()];
        for row in 0..k {
            for col in 0..n {
                let idx = row * n + col;
                let s = b_scale_t.data[col];
                let zp = b_zp_t.data[col];
                data[idx] = (b_quant.data[idx] - zp) * s;
            }
        }
        Tensor::new(b_quant.dims.clone(), data)
    } else {
        Tensor::new(
            b_quant.dims.clone(),
            dequantize(&b_quant.data, b_scale_t.data[0], b_zp_t.data[0]),
        )
    };

    // Use existing matmul logic
    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qmm_a__".to_string(), a_float);
    tmp_values.insert("__qmm_b__".to_string(), b_float);

    let mut mm_node = node.clone();
    mm_node.op_type = "MatMul".to_string();
    mm_node.input = vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()];
    mm_node.output = vec!["__qmm_y__".to_string()];

    exec_matmul(&mm_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qmm_y__").unwrap();

    let y_quant = quantize_u8(&y_float.data, y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

/// QLinearGlobalAveragePool: inputs x, x_scale, x_zp, y_scale, y_zp
pub fn exec_qlinear_global_avg_pool(
    node: &NodeProto,
    values: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.data[0];
    let x_zp = get_tensor(values, &node.input[2])?.data[0];
    let y_scale = get_tensor(values, &node.input[3])?.data[0];
    let y_zp = get_tensor(values, &node.input[4])?.data[0];

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(&x_quant.data, x_scale, x_zp),
    );

    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qgap_x__".to_string(), x_float);

    let mut gap_node = node.clone();
    gap_node.op_type = "GlobalAveragePool".to_string();
    gap_node.input = vec!["__qgap_x__".to_string()];
    gap_node.output = vec!["__qgap_y__".to_string()];

    exec_global_avg_pool(&gap_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qgap_y__").unwrap();

    let y_quant = quantize_u8(&y_float.data, y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}
