use anyhow::Context;
use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

pub struct MatMulPrecomp {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch_size: usize,
    pub a_batch_stride: usize,
    pub b_batch_stride: usize,
    pub o_batch_stride: usize,
    pub a_broadcasts: bool,
    pub b_broadcasts: bool,
    pub out_dims: [usize; 8],
    pub out_rank: usize,
}

pub struct MatMul {
    pub inputs: Vec<String>,
    pub shape_cache: Dims,
    pub precomp: Option<MatMulPrecomp>,
}

impl MatMul {
    pub fn compute_shapes(a: &[usize], b: &[usize]) -> MatMulPrecomp {
        let a_rank = a.len();
        let b_rank = b.len();
        let m = a[a_rank - 2];
        let k = a[a_rank - 1];
        let n = b[b_rank - 1];

        let batch_dims_a = &a[..a_rank - 2];
        let batch_dims_b = &b[..b_rank - 2];
        let batch_ndim = batch_dims_a.len().max(batch_dims_b.len());
        let mut out_dims = [0usize; 8];
        broadcast_shape_into(batch_dims_a, batch_dims_b, &mut out_dims[..batch_ndim]);
        let batch_size: usize = out_dims[..batch_ndim].iter().product();
        out_dims[batch_ndim] = m;
        out_dims[batch_ndim + 1] = n;
        let out_rank = batch_ndim + 2;

        MatMulPrecomp {
            m,
            k,
            n,
            batch_size,
            a_batch_stride: m * k,
            b_batch_stride: k * n,
            o_batch_stride: m * n,
            a_broadcasts: batch_dims_a.is_empty() || batch_dims_a.iter().product::<usize>() == 1,
            b_broadcasts: batch_dims_b.is_empty() || batch_dims_b.iter().product::<usize>() == 1,
            out_dims,
            out_rank,
        }
    }

    pub fn new(inputs: Vec<String>, initial_shape_a: &[usize], initial_shape_b: &[usize]) -> Self {
        let (shape_cache, precomp) = if initial_shape_a.len() >= 2 && initial_shape_b.len() >= 2 {
            let mut cache = Dims::from_slice(initial_shape_a);
            cache.extend_from_slice(initial_shape_b);
            (
                cache,
                Some(Self::compute_shapes(initial_shape_a, initial_shape_b)),
            )
        } else {
            (Dims::new(), None)
        };

        Self {
            inputs,
            shape_cache,
            precomp,
        }
    }
}

impl MatMul {
    /// Naive scalar matmul — used by QLinearMatMul to avoid BLAS-induced rounding
    /// differences in quantized pipelines.
    pub fn execute_naive(
        &mut self,
        values: &HashMap<String, Tensor>,
        output: &mut Tensor,
    ) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let mut key = Dims::from_slice(&a.dims);
        key.extend_from_slice(&b.dims);
        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == key.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(&a.dims, &b.dims));
                self.shape_cache = key;
                self.precomp.as_ref().expect("just set")
            }
        };

        let a_f = a.floats().context("in MatMul layer")?;
        let b_f = b.floats().context("in MatMul layer")?;
        let total = p.batch_size * p.m * p.n;
        let buf = output.as_mut_f32(total);

        for batch in 0..p.batch_size {
            let a_off = if p.a_broadcasts {
                0
            } else {
                batch * p.a_batch_stride
            };
            let b_off = if p.b_broadcasts {
                0
            } else {
                batch * p.b_batch_stride
            };
            let o_off = batch * p.o_batch_stride;
            matmul_naive(
                &a_f[a_off..],
                &b_f[b_off..],
                &mut buf[o_off..],
                p.m,
                p.k,
                p.n,
            );
        }

        output.set_dims(&p.out_dims[..p.out_rank]);
        Ok(())
    }
}

impl Layer for MatMul {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let mut key = Dims::from_slice(&a.dims);
        key.extend_from_slice(&b.dims);
        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == key.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(&a.dims, &b.dims));
                self.shape_cache = key;
                self.precomp.as_ref().expect("just set")
            }
        };

        let a_f = a.floats().context("in MatMul layer")?;
        let b_f = b.floats().context("in MatMul layer")?;
        let total = p.batch_size * p.m * p.n;
        let buf = output.as_mut_f32(total);
        buf.fill(0.0);

        for batch in 0..p.batch_size {
            let a_off = if p.a_broadcasts {
                0
            } else {
                batch * p.a_batch_stride
            };
            let b_off = if p.b_broadcasts {
                0
            } else {
                batch * p.b_batch_stride
            };
            let o_off = batch * p.o_batch_stride;
            crate::blas::sgemm(
                p.m,
                p.n,
                p.k,
                1.0,
                &a_f[a_off..],
                p.k,
                false,
                &b_f[b_off..],
                p.n,
                false,
                0.0,
                &mut buf[o_off..],
                p.n,
            );
        }

        output.set_dims(&p.out_dims[..p.out_rank]);
        Ok(())
    }
}

/// Naive reference implementation for correctness testing.
pub fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for pk in 0..k {
                sum += a[i * k + pk] * b[pk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
