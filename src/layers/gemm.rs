use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

pub struct GemmPrecomp {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

pub struct Gemm {
    pub inputs: Vec<String>,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
    pub shape_cache: Dims,
    pub precomp: Option<GemmPrecomp>,
}

impl Gemm {
    pub fn compute_shapes(
        a_shape: &[usize],
        b_shape: &[usize],
        trans_a: bool,
        trans_b: bool,
    ) -> GemmPrecomp {
        let (m, k) = if trans_a {
            (a_shape[1], a_shape[0])
        } else {
            (a_shape[0], a_shape[1])
        };
        let n = if trans_b { b_shape[0] } else { b_shape[1] };
        GemmPrecomp { m, k, n }
    }

    pub fn new(
        inputs: Vec<String>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
        initial_shape_a: &[usize],
        initial_shape_b: &[usize],
    ) -> Self {
        let (shape_cache, precomp) = if initial_shape_a.len() == 2 && initial_shape_b.len() == 2 {
            let mut cache = Dims::from_slice(initial_shape_a);
            cache.extend_from_slice(initial_shape_b);
            (
                cache,
                Some(Self::compute_shapes(
                    initial_shape_a,
                    initial_shape_b,
                    trans_a,
                    trans_b,
                )),
            )
        } else {
            (Dims::new(), None)
        };

        Self {
            inputs,
            alpha,
            beta,
            trans_a,
            trans_b,
            shape_cache,
            precomp,
        }
    }
}

impl Layer for Gemm {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;
        let c_tensor = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            Some(get_tensor(values, &self.inputs[2])?)
        } else {
            None
        };

        let mut key = Dims::from_slice(&a.dims);
        key.extend_from_slice(&b.dims);
        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == key.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(
                    &a.dims,
                    &b.dims,
                    self.trans_a,
                    self.trans_b,
                ));
                self.shape_cache = key;
                self.precomp.as_ref().expect("just set")
            }
        };

        let m = p.m;
        let k = p.k;
        let n = p.n;

        let a_f = a.floats().context("in Gemm layer")?;
        let b_f = b.floats().context("in Gemm layer")?;
        let buf = output.as_mut_f32(m * n);

        // If we have a bias, pre-fill the output with beta*C, then accumulate with sgemm
        if let Some(c_tensor) = c_tensor {
            let c_f = c_tensor.floats().context("in Gemm layer")?;
            if c_tensor.dims.as_slice() == [m, n] {
                // C is already m x n, copy directly
                buf[..m * n].copy_from_slice(&c_f[..m * n]);
            } else {
                // Broadcast C into buf
                let ndim = 2.max(c_tensor.dims.len());
                let mut c_shape = [0usize; 8];
                broadcast_shape_into(&[m, n], &c_tensor.dims, &mut c_shape[..ndim]);
                for i in 0..m {
                    for j in 0..n {
                        let idx = [i, j];
                        let ci = broadcast_index(&idx, &c_tensor.dims, &c_shape[..ndim]);
                        buf[i * n + j] = c_f[ci];
                    }
                }
            }

            let lda = if self.trans_a { m } else { k };
            let ldb = if self.trans_b { k } else { n };
            crate::blas::sgemm(
                m,
                n,
                k,
                self.alpha,
                a_f,
                lda,
                self.trans_a,
                b_f,
                ldb,
                self.trans_b,
                self.beta,
                buf,
                n,
            );
        } else {
            buf.fill(0.0);
            let lda = if self.trans_a { m } else { k };
            let ldb = if self.trans_b { k } else { n };
            crate::blas::sgemm(
                m,
                n,
                k,
                self.alpha,
                a_f,
                lda,
                self.trans_a,
                b_f,
                ldb,
                self.trans_b,
                0.0,
                buf,
                n,
            );
        }

        output.set_dims(&[m, n]);
        Ok(())
    }
}
