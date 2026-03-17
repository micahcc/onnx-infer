use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::conv::Conv;

pub struct QLinearConv {
    pub inputs: Vec<String>,
    pub inner: Conv,
    /// Reusable i16 buffer for im2col output (2x smaller than f32)
    col_buf: Vec<i16>,
    /// Reusable i16 buffer for weight (after zero-point subtraction)
    w_buf: Vec<i16>,
    /// Reusable i32 buffer for GEMM accumulation
    gemm_buf: Vec<i32>,
}

impl QLinearConv {
    pub fn new(inputs: Vec<String>, inner: Conv) -> Self {
        Self {
            inputs,
            inner,
            col_buf: Vec::new(),
            w_buf: Vec::new(),
            gemm_buf: Vec::new(),
        }
    }
}

/// im2col for i16: unfold input patches, converting f32-encoded quantized values
/// to i16 by subtracting zero_point. Range -255..255 fits safely in i16.
fn im2col_i16(
    input: &[f32],
    col: &mut [i16],
    zero_point: f32,
    c_in_per_group: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    p0: usize,
    p1: usize,
    h_out: usize,
    w_out: usize,
) {
    let zp = zero_point.round() as i16;
    let spatial_out = h_out * w_out;
    for ic in 0..c_in_per_group {
        let in_plane = &input[ic * h_in * w_in..];
        for fh in 0..kh {
            for fw in 0..kw {
                let col_row = (ic * kh + fh) * kw + fw;
                let col_row_off = col_row * spatial_out;
                for oh in 0..h_out {
                    let ih = oh * sh + fh * dh;
                    let valid_h = ih >= p0 && ih - p0 < h_in;
                    let ih_actual = ih.wrapping_sub(p0);
                    for ow in 0..w_out {
                        let iw = ow * sw + fw * dw;
                        let col_idx = col_row_off + oh * w_out + ow;
                        if valid_h && iw >= p1 && iw - p1 < w_in {
                            col[col_idx] =
                                in_plane[ih_actual * w_in + (iw - p1)] as i16 - zp;
                        } else {
                            col[col_idx] = 0;
                        }
                    }
                }
            }
        }
    }
}

impl Layer for QLinearConv {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_quant = get_tensor(values, &self.inputs[0])?;
        let x_scale = get_tensor(values, &self.inputs[1])?.floats()[0];
        let x_zp = get_tensor(values, &self.inputs[2])?.floats()[0];
        let w_quant = get_tensor(values, &self.inputs[3])?;
        let w_scale_t = get_tensor(values, &self.inputs[4])?;
        let w_zp_t = get_tensor(values, &self.inputs[5])?;
        let y_scale = get_tensor(values, &self.inputs[6])?.floats()[0];
        let y_zp = get_tensor(values, &self.inputs[7])?.floats()[0];
        let bias = if self.inputs.len() > 8 && !self.inputs[8].is_empty() {
            Some(get_tensor(values, &self.inputs[8])?)
        } else {
            None
        };

        // Resolve conv shapes
        let p = {
            let shape_matches = match &self.inner.precomp {
                Some(_) => self.inner.shape_cache.as_slice() == x_quant.dims.as_slice(),
                None => false,
            };
            if !shape_matches {
                self.inner.precomp = Some(Conv::compute_shapes(
                    &x_quant.dims,
                    &w_quant.dims,
                    self.inner.kh,
                    self.inner.kw,
                    self.inner.sh,
                    self.inner.sw,
                    self.inner.dh,
                    self.inner.dw,
                    self.inner.auto_pad,
                    &self.inner.pads,
                    self.inner.group,
                ));
                self.inner.shape_cache.clone_from(&x_quant.dims);
            }
            self.inner.precomp.as_ref().expect("just set")
        };

        let kh = self.inner.kh;
        let kw = self.inner.kw;
        let group = self.inner.group;
        let spatial_out = p.h_out * p.w_out;
        let gemm_m = p.c_out_per_group;
        let gemm_k = p.c_in_per_group * kh * kw;
        let gemm_n = spatial_out;

        // Convert weight f32→i16 (subtract per-channel zero point)
        let w_quant_f = w_quant.floats();
        let w_scale_f = w_scale_t.floats();
        let per_channel = w_scale_f.len() > 1;
        let elems_per_oc = w_quant.numel() / p.c_out;
        self.w_buf.resize(w_quant.numel(), 0);
        for oc in 0..p.c_out {
            let zp = if per_channel {
                w_zp_t.f32_at(oc).round() as i16
            } else {
                w_zp_t.f32_at(0).round() as i16
            };
            let base = oc * elems_per_oc;
            for i in 0..elems_per_oc {
                self.w_buf[base + i] = w_quant_f[base + i] as i16 - zp;
            }
        }

        // Prepare im2col and gemm buffers
        let col_size = gemm_k * gemm_n;
        self.col_buf.resize(col_size, 0);
        self.gemm_buf.resize(gemm_m * gemm_n, 0);

        let x_quant_f = x_quant.floats();
        let total = p.n * p.c_out * spatial_out;
        let buf = output.as_mut_f32(total);
        let inv_y_scale = 1.0 / y_scale;

        for batch in 0..p.n {
            let in_batch = batch * p.c_in * p.h_in * p.w_in;
            for g in 0..group {
                let in_group = in_batch + g * p.c_in_per_group * p.h_in * p.w_in;

                im2col_i16(
                    &x_quant_f[in_group..],
                    &mut self.col_buf,
                    x_zp,
                    p.c_in_per_group,
                    p.h_in,
                    p.w_in,
                    kh,
                    kw,
                    self.inner.sh,
                    self.inner.sw,
                    self.inner.dh,
                    self.inner.dw,
                    p.p0,
                    p.p1,
                    p.h_out,
                    p.w_out,
                );

                // i16 GEMM: gemm_buf[m,n] = W_i16[m,k] * col_i16[k,n] (accumulated in i32)
                let w_group = g * p.c_out_per_group * gemm_k;
                self.gemm_buf.fill(0);
                crate::blas::i16_gemm(
                    gemm_m,
                    gemm_n,
                    gemm_k,
                    &self.w_buf[w_group..],
                    gemm_k,
                    &self.col_buf,
                    gemm_n,
                    &mut self.gemm_buf,
                    gemm_n,
                );

                // Scale i32 → f32 and quantize output
                let out_group =
                    batch * p.c_out * spatial_out + g * p.c_out_per_group * spatial_out;
                for oc in 0..gemm_m {
                    let abs_oc = g * p.c_out_per_group + oc;
                    let combined_scale = x_scale
                        * if per_channel {
                            w_scale_f[abs_oc]
                        } else {
                            w_scale_f[0]
                        };
                    let bias_val = if let Some(b) = bias {
                        let ws = if per_channel {
                            w_scale_f[abs_oc]
                        } else {
                            w_scale_f[0]
                        };
                        b.f32_at(abs_oc) * x_scale * ws
                    } else {
                        0.0
                    };
                    let row_start = out_group + oc * spatial_out;
                    let gemm_row = &self.gemm_buf[oc * gemm_n..];
                    for s in 0..spatial_out {
                        let float_val = gemm_row[s] as f32 * combined_scale + bias_val;
                        buf[row_start + s] =
                            (float_val * inv_y_scale + y_zp).round().clamp(0.0, 255.0);
                    }
                }
            }
        }

        output.set_dims(&[p.n, p.c_out, p.h_out, p.w_out]);
        Ok(())
    }
}
