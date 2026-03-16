use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

pub struct MatMul {
    pub inputs: Vec<String>,
    // Precomputed (0 = not precomputed)
    pub pre_m: usize,
    pub pre_k: usize,
    pub pre_n: usize,
    pub pre_batch_size: usize,
    pub pre_a_batch_stride: usize,
    pub pre_b_batch_stride: usize,
    pub pre_o_batch_stride: usize,
    pub pre_a_broadcasts: bool,
    pub pre_b_broadcasts: bool,
    pub pre_out_dims: [usize; 8],
    pub pre_out_rank: usize,
}

impl MatMul {
    pub fn new(inputs: Vec<String>, a_shape: &[usize], b_shape: &[usize]) -> Self {
        let mut s = Self {
            inputs,
            pre_m: 0,
            pre_k: 0,
            pre_n: 0,
            pre_batch_size: 0,
            pre_a_batch_stride: 0,
            pre_b_batch_stride: 0,
            pre_o_batch_stride: 0,
            pre_a_broadcasts: false,
            pre_b_broadcasts: false,
            pre_out_dims: [0; 8],
            pre_out_rank: 0,
        };

        if a_shape.len() >= 2 && b_shape.len() >= 2 {
            s.precompute(a_shape, b_shape);
        }
        s
    }

    fn precompute(&mut self, a: &[usize], b: &[usize]) {
        let a_rank = a.len();
        let b_rank = b.len();
        let m = a[a_rank - 2];
        let k = a[a_rank - 1];
        let n = b[b_rank - 1];

        let batch_dims_a = &a[..a_rank - 2];
        let batch_dims_b = &b[..b_rank - 2];
        let batch_ndim = batch_dims_a.len().max(batch_dims_b.len());
        broadcast_shape_into(
            batch_dims_a,
            batch_dims_b,
            &mut self.pre_out_dims[..batch_ndim],
        );
        let batch_size: usize = self.pre_out_dims[..batch_ndim].iter().product();
        self.pre_out_dims[batch_ndim] = m;
        self.pre_out_dims[batch_ndim + 1] = n;
        self.pre_out_rank = batch_ndim + 2;

        self.pre_m = m;
        self.pre_k = k;
        self.pre_n = n;
        self.pre_batch_size = batch_size;
        self.pre_a_batch_stride = m * k;
        self.pre_b_batch_stride = k * n;
        self.pre_o_batch_stride = m * n;
        self.pre_a_broadcasts =
            batch_dims_a.is_empty() || batch_dims_a.iter().product::<usize>() == 1;
        self.pre_b_broadcasts =
            batch_dims_b.is_empty() || batch_dims_b.iter().product::<usize>() == 1;
    }
}

impl Layer for MatMul {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let (
            m,
            k,
            n,
            batch_size,
            a_batch_stride,
            b_batch_stride,
            o_batch_stride,
            a_broadcasts,
            b_broadcasts,
            out_rank,
        ) = if self.pre_out_rank > 0 {
            (
                self.pre_m,
                self.pre_k,
                self.pre_n,
                self.pre_batch_size,
                self.pre_a_batch_stride,
                self.pre_b_batch_stride,
                self.pre_o_batch_stride,
                self.pre_a_broadcasts,
                self.pre_b_broadcasts,
                self.pre_out_rank,
            )
        } else {
            let a_rank = a.dims.len();
            let b_rank = b.dims.len();
            let m = a.dims[a_rank - 2];
            let k = a.dims[a_rank - 1];
            let n = b.dims[b_rank - 1];
            let batch_dims_a = &a.dims[..a_rank - 2];
            let batch_dims_b = &b.dims[..b_rank - 2];
            let batch_ndim = batch_dims_a.len().max(batch_dims_b.len());
            broadcast_shape_into(
                batch_dims_a,
                batch_dims_b,
                &mut self.pre_out_dims[..batch_ndim],
            );
            let batch_size: usize = self.pre_out_dims[..batch_ndim].iter().product();
            self.pre_out_dims[batch_ndim] = m;
            self.pre_out_dims[batch_ndim + 1] = n;
            let out_rank = batch_ndim + 2;
            let a_broadcasts =
                batch_dims_a.is_empty() || batch_dims_a.iter().product::<usize>() == 1;
            let b_broadcasts =
                batch_dims_b.is_empty() || batch_dims_b.iter().product::<usize>() == 1;
            (
                m,
                k,
                n,
                batch_size,
                m * k,
                k * n,
                m * n,
                a_broadcasts,
                b_broadcasts,
                out_rank,
            )
        };

        let a_f = a.floats();
        let b_f = b.floats();
        let total = batch_size * m * n;
        let buf = output.as_mut_f32(total);
        buf.fill(0.0);

        for batch in 0..batch_size {
            let a_off = if a_broadcasts {
                0
            } else {
                batch * a_batch_stride
            };
            let b_off = if b_broadcasts {
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

        output.set_dims(&self.pre_out_dims[..out_rank]);
        Ok(())
    }
}
