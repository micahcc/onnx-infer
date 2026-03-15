use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_shape;
use crate::get_tensor;
use crate::layers::Layer;

pub struct MatMul {
    pub inputs: Vec<String>,
}

impl MatMul {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for MatMul {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let a_rank = a.dims.len();
        let b_rank = b.dims.len();

        let m = a.dims[a_rank - 2];
        let k = a.dims[a_rank - 1];
        let n = b.dims[b_rank - 1];

        let batch_dims_a = &a.dims[..a_rank - 2];
        let batch_dims_b = &b.dims[..b_rank - 2];
        let batch_shape = broadcast_shape(batch_dims_a, batch_dims_b);
        let batch_size: usize = batch_shape.iter().product();

        let mut out_dims = batch_shape.clone();
        out_dims.push(m);
        out_dims.push(n);

        let a_f = a.floats();
        let b_f = b.floats();
        let total = batch_size * m * n;
        let buf = output.as_mut_f32(total);
        buf.fill(0.0);

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
                        sum += a_f[a_off + i * k + p] * b_f[b_off + p * n + j];
                    }
                    buf[batch * o_batch_stride + i * n + j] = sum;
                }
            }
        }

        output.dims = out_dims;
        Ok(())
    }
}
