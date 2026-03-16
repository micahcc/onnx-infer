use std::collections::HashMap;

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
    pub precomp: Option<GemmPrecomp>,
}

impl Gemm {
    pub fn new(
        inputs: Vec<String>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Self {
        let precomp = if a_shape.len() == 2 && b_shape.len() == 2 {
            {
                let (a, b) = (a_shape, b_shape);
                let (m, k) = if trans_a { (a[1], a[0]) } else { (a[0], a[1]) };
                let n = if trans_b { b[0] } else { b[1] };
                Some(GemmPrecomp { m, k, n })
            }
        } else {
            None
        };

        Self {
            inputs,
            alpha,
            beta,
            trans_a,
            trans_b,
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

        let (m, k, n) = if let Some(p) = &self.precomp {
            (p.m, p.k, p.n)
        } else {
            let (m, k) = if self.trans_a {
                (a.dims[1], a.dims[0])
            } else {
                (a.dims[0], a.dims[1])
            };
            let n = if self.trans_b { b.dims[0] } else { b.dims[1] };
            (m, k, n)
        };

        let a_f = a.floats();
        let b_f = b.floats();
        let buf = output.as_mut_f32(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = if self.trans_a {
                        a_f[p * m + i]
                    } else {
                        a_f[i * k + p]
                    };
                    let b_val = if self.trans_b {
                        b_f[j * k + p]
                    } else {
                        b_f[p * n + j]
                    };
                    sum += a_val * b_val;
                }
                buf[i * n + j] = self.alpha * sum;
            }
        }

        if let Some(c_tensor) = c_tensor {
            let c_f = c_tensor.floats();
            let ndim = 2.max(c_tensor.dims.len());
            let mut c_shape = [0usize; 8];
            broadcast_shape_into(&[m, n], &c_tensor.dims, &mut c_shape[..ndim]);
            for i in 0..m {
                for j in 0..n {
                    let idx = [i, j];
                    let ci = broadcast_index(&idx, &c_tensor.dims, &c_shape[..ndim]);
                    buf[i * n + j] += self.beta * c_f[ci];
                }
            }
        }

        output.set_dims(&[m, n]);
        Ok(())
    }
}
