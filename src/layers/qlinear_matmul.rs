use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::matmul::MatMul;

pub struct QLinearMatMul {
    pub inputs: Vec<String>,
    pub inner: MatMul,
    /// Reusable i16 buffers (2x smaller than f32)
    a_buf: Vec<i16>,
    b_buf: Vec<i16>,
    c_buf: Vec<i32>,
}

impl QLinearMatMul {
    pub fn new(inputs: Vec<String>, inner: MatMul) -> Self {
        Self {
            inputs,
            inner,
            a_buf: Vec::new(),
            b_buf: Vec::new(),
            c_buf: Vec::new(),
        }
    }
}

impl Layer for QLinearMatMul {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a_quant = get_tensor(values, &self.inputs[0])?;
        let a_scale = get_tensor(values, &self.inputs[1])?.floats().context("in QLinearMatMul layer")?[0];
        let a_zp = get_tensor(values, &self.inputs[2])?.floats().context("in QLinearMatMul layer")?[0].round() as i16;
        let b_quant = get_tensor(values, &self.inputs[3])?;
        let b_scale_t = get_tensor(values, &self.inputs[4])?;
        let b_zp_t = get_tensor(values, &self.inputs[5])?;
        let y_scale = get_tensor(values, &self.inputs[6])?.floats().context("in QLinearMatMul layer")?[0];
        let y_zp = get_tensor(values, &self.inputs[7])?.floats().context("in QLinearMatMul layer")?[0];

        // Resolve matmul shapes
        use crate::Dims;
        let mut key = Dims::from_slice(&a_quant.dims);
        key.extend_from_slice(&b_quant.dims);
        let p = match &self.inner.precomp {
            Some(p) if self.inner.shape_cache.as_slice() == key.as_slice() => p,
            _ => {
                self.inner.precomp = Some(MatMul::compute_shapes(&a_quant.dims, &b_quant.dims));
                self.inner.shape_cache = key;
                self.inner.precomp.as_ref().expect("just set")
            }
        };

        let m = p.m;
        let k = p.k;
        let n = p.n;

        // Convert A f32→i16 (subtract zero point)
        let a_f = a_quant.floats().context("in QLinearMatMul layer")?;
        self.a_buf.resize(a_f.len(), 0);
        for (o, &v) in self.a_buf.iter_mut().zip(a_f.iter()) {
            *o = v as i16 - a_zp;
        }

        // Convert B f32→i16 (subtract per-column or scalar zero point)
        let b_f = b_quant.floats().context("in QLinearMatMul layer")?;
        let b_scale_f = b_scale_t.floats().context("in QLinearMatMul layer")?;
        let per_column = b_scale_f.len() > 1;
        self.b_buf.resize(b_f.len(), 0);
        if per_column {
            let rows = b_f.len() / n;
            for row in 0..rows {
                for col in 0..n {
                    let idx = row * n + col;
                    self.b_buf[idx] = b_f[idx] as i16 - b_zp_t.f32_at(col).context("in QLinearMatMul layer")?.round() as i16;
                }
            }
        } else {
            let b_zp = b_zp_t.f32_at(0).context("in QLinearMatMul layer")?.round() as i16;
            for (o, &v) in self.b_buf.iter_mut().zip(b_f.iter()) {
                *o = v as i16 - b_zp;
            }
        }

        // i16 GEMM per batch, accumulated in i32
        let total = p.batch_size * m * n;
        self.c_buf.resize(m * n, 0);
        let buf = output.as_mut_f32(total);
        let inv_y_scale = 1.0 / y_scale;

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

            self.c_buf.fill(0);
            crate::blas::i16_gemm(
                m,
                n,
                k,
                &self.a_buf[a_off..],
                k,
                &self.b_buf[b_off..],
                n,
                &mut self.c_buf,
                n,
            );

            // Scale i32 → f32 and quantize
            if per_column {
                for i in 0..m {
                    for j in 0..n {
                        let combined_scale = a_scale * b_scale_f[j];
                        let float_val = self.c_buf[i * n + j] as f32 * combined_scale;
                        buf[o_off + i * n + j] =
                            (float_val * inv_y_scale + y_zp).round().clamp(0.0, 255.0);
                    }
                }
            } else {
                let combined_scale = a_scale * b_scale_f[0];
                for idx in 0..m * n {
                    let float_val = self.c_buf[idx] as f32 * combined_scale;
                    buf[o_off + idx] = (float_val * inv_y_scale + y_zp).round().clamp(0.0, 255.0);
                }
            }
        }

        output.set_dims(&p.out_dims[..p.out_rank]);
        Ok(())
    }
}
