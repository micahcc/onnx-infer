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
        get_tensor(values, &node.input[2])?.floats()[0]
    } else {
        0.0
    };
    let data = quantize_u8(input.floats(), scale.floats()[0], zero_point);
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

    let input_f = input.to_f32_vec();
    let scale_f = scale.floats();
    let zp_f = zero_point.to_f32_vec();

    if scale_f.len() == 1 {
        let data = dequantize(&input_f, scale_f[0], zp_f[0]);
        values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    } else {
        let outer: usize = input.dims[..axis].iter().product();
        let ch = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();
        let mut data = vec![0.0f32; input_f.len()];
        for o in 0..outer {
            for c in 0..ch {
                let s = scale_f[c];
                let zp = zp_f[c];
                let base = (o * ch + c) * inner;
                for i in 0..inner {
                    data[base + i] = (input_f[base + i] - zp) * s;
                }
            }
        }
        values.insert(node.output[0].clone(), Tensor::new(input.dims, data));
    }
    Ok(())
}

pub fn exec_qlinear_conv(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.floats()[0];
    let x_zp = get_tensor(values, &node.input[2])?.floats()[0];
    let w_quant = get_tensor(values, &node.input[3])?;
    let w_scale_t = get_tensor(values, &node.input[4])?;
    let w_zp_t = get_tensor(values, &node.input[5])?;
    let y_scale = get_tensor(values, &node.input[6])?.floats()[0];
    let y_zp = get_tensor(values, &node.input[7])?.floats()[0];
    let bias = if node.input.len() > 8 && !node.input[8].is_empty() {
        Some(get_tensor(values, &node.input[8])?)
    } else {
        None
    };

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(x_quant.floats(), x_scale, x_zp),
    );

    let c_out = w_quant.dims[0];
    let w_scale_f = w_scale_t.floats();
    let w_zp_f = w_zp_t.to_f32_vec();
    let per_channel = w_scale_f.len() > 1;
    let elems_per_oc = w_quant.numel() / c_out;
    let w_quant_f = w_quant.floats();
    let mut w_float_data = vec![0.0f32; w_quant.numel()];
    for oc in 0..c_out {
        let scale = if per_channel {
            w_scale_f[oc]
        } else {
            w_scale_f[0]
        };
        let zp = if per_channel { w_zp_f[oc] } else { w_zp_f[0] };
        let base = oc * elems_per_oc;
        for i in 0..elems_per_oc {
            w_float_data[base + i] = (w_quant_f[base + i] - zp) * scale;
        }
    }
    let w_float = Tensor::new(w_quant.dims.clone(), w_float_data);

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
        // Bias is int32 (now stored as Int64). Convert to float.
        let b_f = b.to_f32_vec();
        let bias_float: Vec<f32> = b_f
            .iter()
            .enumerate()
            .map(|(oc, &val)| {
                let ws = if per_channel {
                    w_scale_f[oc]
                } else {
                    w_scale_f[0]
                };
                val * x_scale * ws
            })
            .collect();
        tmp_values.insert("__qconv_b__".to_string(), Tensor::new(b.dims, bias_float));
    }

    exec_conv(&conv_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qconv_y__").unwrap();

    let y_quant = quantize_u8(y_float.floats(), y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

pub fn exec_qlinear_add(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.floats()[0];
    let x_zp = get_tensor(values, &node.input[2])?.floats()[0];
    let y_quant = get_tensor(values, &node.input[3])?;
    let y_scale = get_tensor(values, &node.input[4])?.floats()[0];
    let y_zp = get_tensor(values, &node.input[5])?.floats()[0];
    let z_scale = get_tensor(values, &node.input[6])?.floats()[0];
    let z_zp = get_tensor(values, &node.input[7])?.floats()[0];

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(x_quant.floats(), x_scale, x_zp),
    );
    let y_float = Tensor::new(
        y_quant.dims.clone(),
        dequantize(y_quant.floats(), y_scale, y_zp),
    );

    let out_shape = broadcast_shape(&x_float.dims, &y_float.dims);
    let numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut index = vec![0usize; ndim];
    let x_f = x_float.floats();
    let y_f = y_float.floats();
    let mut z_float = vec![0.0f32; numel];

    for val in &mut z_float {
        let ai = broadcast_index(&index, &x_float.dims, &out_shape);
        let bi = broadcast_index(&index, &y_float.dims, &out_shape);
        *val = x_f[ai] + y_f[bi];
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

pub fn exec_qlinear_matmul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a_quant = get_tensor(values, &node.input[0])?;
    let a_scale = get_tensor(values, &node.input[1])?.floats()[0];
    let a_zp = get_tensor(values, &node.input[2])?.floats()[0];
    let b_quant = get_tensor(values, &node.input[3])?;
    let b_scale_t = get_tensor(values, &node.input[4])?;
    let b_zp_t = get_tensor(values, &node.input[5])?;
    let y_scale = get_tensor(values, &node.input[6])?.floats()[0];
    let y_zp = get_tensor(values, &node.input[7])?.floats()[0];

    let a_float = Tensor::new(
        a_quant.dims.clone(),
        dequantize(a_quant.floats(), a_scale, a_zp),
    );

    let b_scale_f = b_scale_t.floats();
    let b_zp_f = b_zp_t.to_f32_vec();
    let b_quant_f = b_quant.floats();
    let b_float = if b_scale_f.len() > 1 {
        let n = *b_quant.dims.last().unwrap();
        let k = b_quant.numel() / n;
        let mut data = vec![0.0f32; b_quant.numel()];
        for row in 0..k {
            for col in 0..n {
                let idx = row * n + col;
                data[idx] = (b_quant_f[idx] - b_zp_f[col]) * b_scale_f[col];
            }
        }
        Tensor::new(b_quant.dims.clone(), data)
    } else {
        Tensor::new(
            b_quant.dims.clone(),
            dequantize(b_quant_f, b_scale_f[0], b_zp_f[0]),
        )
    };

    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qmm_a__".to_string(), a_float);
    tmp_values.insert("__qmm_b__".to_string(), b_float);

    let mut mm_node = node.clone();
    mm_node.op_type = "MatMul".to_string();
    mm_node.input = vec!["__qmm_a__".to_string(), "__qmm_b__".to_string()];
    mm_node.output = vec!["__qmm_y__".to_string()];

    exec_matmul(&mm_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qmm_y__").unwrap();

    let y_quant = quantize_u8(y_float.floats(), y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}

pub fn exec_qlinear_global_avg_pool(
    node: &NodeProto,
    values: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let x_quant = get_tensor(values, &node.input[0])?;
    let x_scale = get_tensor(values, &node.input[1])?.floats()[0];
    let x_zp = get_tensor(values, &node.input[2])?.floats()[0];
    let y_scale = get_tensor(values, &node.input[3])?.floats()[0];
    let y_zp = get_tensor(values, &node.input[4])?.floats()[0];

    let x_float = Tensor::new(
        x_quant.dims.clone(),
        dequantize(x_quant.floats(), x_scale, x_zp),
    );

    let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
    tmp_values.insert("__qgap_x__".to_string(), x_float);

    let mut gap_node = node.clone();
    gap_node.op_type = "GlobalAveragePool".to_string();
    gap_node.input = vec!["__qgap_x__".to_string()];
    gap_node.output = vec!["__qgap_y__".to_string()];

    exec_global_avg_pool(&gap_node, &mut tmp_values)?;
    let y_float = tmp_values.remove("__qgap_y__").unwrap();

    let y_quant = quantize_u8(y_float.floats(), y_scale, y_zp);
    values.insert(node.output[0].clone(), Tensor::new(y_float.dims, y_quant));
    Ok(())
}
