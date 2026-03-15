use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape;
use crate::get_attr_float;
use crate::get_attr_int;
use crate::get_tensor;
use crate::onnx::NodeProto;

pub fn exec_matmul(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let b = get_tensor(values, &node.input[1])?;

    // Support 2D matmul: [M, K] x [K, N] -> [M, N]
    // Also handle batched: [..., M, K] x [..., K, N] -> [..., M, N]
    let a_rank = a.dims.len();
    let b_rank = b.dims.len();

    let m = a.dims[a_rank - 2];
    let k = a.dims[a_rank - 1];
    let n = b.dims[b_rank - 1];

    // For simplicity, handle 2D case and batch prefix
    let batch_dims_a = &a.dims[..a_rank - 2];
    let batch_dims_b = &b.dims[..b_rank - 2];
    let batch_shape = broadcast_shape(batch_dims_a, batch_dims_b);
    let batch_size: usize = batch_shape.iter().product();

    let mut out_dims = batch_shape.clone();
    out_dims.push(m);
    out_dims.push(n);

    let mut output = vec![0.0f32; batch_size * m * n];

    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let o_batch_stride = m * n;

    for batch in 0..batch_size {
        let a_off = if batch_dims_a.is_empty() || batch_dims_a.iter().product::<usize>() == 1 {
            0
        } else {
            batch * a_batch_stride
        };
        let b_off = if batch_dims_b.is_empty() || batch_dims_b.iter().product::<usize>() == 1 {
            0
        } else {
            batch * b_batch_stride
        };
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a.data[a_off + i * k + p] * b.data[b_off + p * n + j];
                }
                output[batch * o_batch_stride + i * n + j] = sum;
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(out_dims, output));
    Ok(())
}

pub fn exec_gemm(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let a = get_tensor(values, &node.input[0])?;
    let b = get_tensor(values, &node.input[1])?;
    let c_tensor = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(get_tensor(values, &node.input[2])?)
    } else {
        None
    };

    let alpha = get_attr_float(node, "alpha").unwrap_or(1.0);
    let beta = get_attr_float(node, "beta").unwrap_or(1.0);
    let trans_a = get_attr_int(node, "transA").unwrap_or(0) != 0;
    let trans_b = get_attr_int(node, "transB").unwrap_or(0) != 0;

    let (m, k_a) = if trans_a {
        (a.dims[1], a.dims[0])
    } else {
        (a.dims[0], a.dims[1])
    };
    let (k_b, n) = if trans_b {
        (b.dims[1], b.dims[0])
    } else {
        (b.dims[0], b.dims[1])
    };
    debug_assert_eq!(k_a, k_b);
    let k = k_a;

    let mut output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a {
                    a.data[p * m + i]
                } else {
                    a.data[i * k + p]
                };
                let b_val = if trans_b {
                    b.data[j * k + p]
                } else {
                    b.data[p * n + j]
                };
                sum += a_val * b_val;
            }
            output[i * n + j] = alpha * sum;
        }
    }

    if let Some(c_tensor) = c_tensor {
        // Broadcast-add C (typically [N] or [M, N])
        let c_shape = broadcast_shape(&[m, n], &c_tensor.dims);
        for i in 0..m {
            for j in 0..n {
                let idx = [i, j];
                let ci = broadcast_index(&idx, &c_tensor.dims, &c_shape);
                output[i * n + j] += beta * c_tensor.data[ci];
            }
        }
    }

    values.insert(node.output[0].clone(), Tensor::new(vec![m, n], output));
    Ok(())
}
