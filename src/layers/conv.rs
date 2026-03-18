use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Clone, Copy)]
pub enum AutoPad {
    Valid,
    SameUpper,
    SameLower,
}

pub struct ConvPrecomp {
    pub n: usize,
    pub c_in: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_out: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub p0: usize,
    pub p1: usize,
    pub c_in_per_group: usize,
    pub c_out_per_group: usize,
    pub total: usize,
}

pub struct Conv {
    pub inputs: Vec<String>,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub dh: usize,
    pub dw: usize,
    pub pads: [usize; 4],
    pub group: usize,
    pub auto_pad: AutoPad,
    pub shape_cache: Dims,
    pub precomp: Option<ConvPrecomp>,
    col_buf: Vec<f32>,
}

impl Conv {
    pub fn compute_shapes(
        ins: &[usize],
        ws: &[usize],
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        dh: usize,
        dw: usize,
        auto_pad: AutoPad,
        pads: &[usize; 4],
        group: usize,
    ) -> ConvPrecomp {
        let n = ins[0];
        let c_in = ins[1];
        let h_in = ins[2];
        let w_in = ins[3];
        let c_out = ws[0];
        let eff_kh = if kh == 0 { ws[2] } else { kh };
        let eff_kw = if kw == 0 { ws[3] } else { kw };

        let (p0, p1, p2, p3) = match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                let oh = h_in.div_ceil(sh);
                let ow = w_in.div_ceil(sw);
                let pad_h = ((oh - 1) * sh + dh * (eff_kh - 1) + 1).saturating_sub(h_in);
                let pad_w = ((ow - 1) * sw + dw * (eff_kw - 1) + 1).saturating_sub(w_in);
                match auto_pad {
                    AutoPad::SameUpper => {
                        (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
                    }
                    _ => (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2),
                }
            }
            AutoPad::Valid => (pads[0], pads[1], pads[2], pads[3]),
        };

        let h_out = (h_in + p0 + p2 - dh * (eff_kh - 1) - 1) / sh + 1;
        let w_out = (w_in + p1 + p3 - dw * (eff_kw - 1) - 1) / sw + 1;

        ConvPrecomp {
            n,
            c_in,
            h_in,
            w_in,
            c_out,
            h_out,
            w_out,
            p0,
            p1,
            c_in_per_group: c_in / group,
            c_out_per_group: c_out / group,
            total: n * c_out * h_out * w_out,
        }
    }

    pub fn new(
        inputs: Vec<String>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: usize,
        auto_pad: String,
        initial_shape: &[usize],
        weight_shape: &[usize],
    ) -> Self {
        let kh = kernel_shape.first().copied().unwrap_or(0) as usize;
        let kw = kernel_shape.get(1).copied().unwrap_or(0) as usize;
        let sh = strides[0] as usize;
        let sw = strides[1] as usize;
        let dh = dilations[0] as usize;
        let dw = dilations[1] as usize;
        let auto_pad_enum = match auto_pad.as_str() {
            "SAME_UPPER" => AutoPad::SameUpper,
            "SAME_LOWER" => AutoPad::SameLower,
            _ => AutoPad::Valid,
        };
        let pads_arr = [
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        ];

        let resolved_kh = if kh == 0 && weight_shape.len() == 4 {
            weight_shape[2]
        } else {
            kh
        };
        let resolved_kw = if kw == 0 && weight_shape.len() == 4 {
            weight_shape[3]
        } else {
            kw
        };

        let (shape_cache, precomp) = if initial_shape.len() == 4 && weight_shape.len() == 4 {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(
                    initial_shape,
                    weight_shape,
                    kh,
                    kw,
                    sh,
                    sw,
                    dh,
                    dw,
                    auto_pad_enum,
                    &pads_arr,
                    group,
                )),
            )
        } else {
            (Dims::new(), None)
        };

        Self {
            inputs,
            kh: resolved_kh,
            kw: resolved_kw,
            sh,
            sw,
            dh,
            dw,
            pads: pads_arr,
            group,
            auto_pad: auto_pad_enum,
            shape_cache,
            precomp,
            col_buf: Vec::new(),
        }
    }
}

/// im2col: unfold input patches into a column matrix.
///
/// For each output pixel (oh, ow), collect the c_in_per_group * kh * kw input values
/// that contribute to it. The result is a matrix of shape
/// [c_in_per_group * kh * kw, h_out * w_out] (column-major-ish: each column is one patch).
fn im2col(
    input: &[f32],
    col: &mut [f32],
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
                            col[col_idx] = in_plane[ih_actual * w_in + (iw - p1)];
                        } else {
                            col[col_idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

impl Conv {
    /// Naive scalar convolution — matches the original accumulation order.
    /// Used by QLinearConv to avoid BLAS-induced rounding differences in quantized pipelines.
    pub fn execute_naive(
        &mut self,
        values: &HashMap<String, Tensor>,
        output: &mut Tensor,
    ) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let weight = get_tensor(values, &self.inputs[1])?;

        let bias = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            Some(get_tensor(values, &self.inputs[2])?)
        } else {
            None
        };

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(
                    &input.dims,
                    &weight.dims,
                    self.kh,
                    self.kw,
                    self.sh,
                    self.sw,
                    self.dh,
                    self.dw,
                    self.auto_pad,
                    &self.pads,
                    self.group,
                ));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let input_f = input.floats();
        let weight_f = weight.floats();
        let buf = output.as_mut_f32(p.total);
        conv_naive(
            input_f,
            weight_f,
            bias.map(|b| b.floats()),
            buf,
            p.n,
            p.c_in,
            p.h_in,
            p.w_in,
            p.c_out,
            self.kh,
            self.kw,
            self.sh,
            self.sw,
            self.dh,
            self.dw,
            p.p0,
            p.p1,
            p.h_out,
            p.w_out,
            self.group,
        );
        output.set_dims(&[p.n, p.c_out, p.h_out, p.w_out]);
        Ok(())
    }
}

impl Layer for Conv {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let weight = get_tensor(values, &self.inputs[1])?;

        let bias = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            Some(get_tensor(values, &self.inputs[2])?)
        } else {
            None
        };

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(
                    &input.dims,
                    &weight.dims,
                    self.kh,
                    self.kw,
                    self.sh,
                    self.sw,
                    self.dh,
                    self.dw,
                    self.auto_pad,
                    &self.pads,
                    self.group,
                ));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let kh = self.kh;
        let kw = self.kw;
        let sh = self.sh;
        let sw = self.sw;
        let dh = self.dh;
        let dw = self.dw;
        let spatial_out = p.h_out * p.w_out;

        let input_f = input.floats();
        let weight_f = weight.floats();
        let buf = output.as_mut_f32(p.total);

        // im2col + sgemm approach
        // For each group: weight is [c_out_per_group, c_in_per_group * kh * kw]
        //                 col is [c_in_per_group * kh * kw, h_out * w_out]
        //                 output is [c_out_per_group, h_out * w_out]
        let col_size = p.c_in_per_group * kh * kw * spatial_out;
        self.col_buf.resize(col_size, 0.0);

        let gemm_m = p.c_out_per_group;
        let gemm_k = p.c_in_per_group * kh * kw;
        let gemm_n = spatial_out;

        for batch in 0..p.n {
            let in_batch = batch * p.c_in * p.h_in * p.w_in;
            for g in 0..self.group {
                let in_group = in_batch + g * p.c_in_per_group * p.h_in * p.w_in;

                im2col(
                    &input_f[in_group..],
                    &mut self.col_buf,
                    p.c_in_per_group,
                    p.h_in,
                    p.w_in,
                    kh,
                    kw,
                    sh,
                    sw,
                    dh,
                    dw,
                    p.p0,
                    p.p1,
                    p.h_out,
                    p.w_out,
                );

                let w_group = g * p.c_out_per_group * gemm_k;
                let out_group = batch * p.c_out * spatial_out + g * p.c_out_per_group * spatial_out;

                // Fill output with bias if present
                if let Some(bias) = bias {
                    let bias_f = bias.floats();
                    for oc in 0..gemm_m {
                        let abs_oc = g * p.c_out_per_group + oc;
                        let row_start = out_group + oc * spatial_out;
                        for s in 0..spatial_out {
                            buf[row_start + s] = bias_f[abs_oc];
                        }
                    }
                    crate::blas::sgemm(
                        gemm_m,
                        gemm_n,
                        gemm_k,
                        1.0,
                        &weight_f[w_group..],
                        gemm_k,
                        false,
                        &self.col_buf,
                        gemm_n,
                        false,
                        1.0,
                        &mut buf[out_group..],
                        gemm_n,
                    );
                } else {
                    crate::blas::sgemm(
                        gemm_m,
                        gemm_n,
                        gemm_k,
                        1.0,
                        &weight_f[w_group..],
                        gemm_k,
                        false,
                        &self.col_buf,
                        gemm_n,
                        false,
                        0.0,
                        &mut buf[out_group..],
                        gemm_n,
                    );
                }
            }
        }

        output.set_dims(&[p.n, p.c_out, p.h_out, p.w_out]);
        Ok(())
    }
}

/// Naive reference implementation for correctness testing.
pub fn conv_naive(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
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
    group: usize,
) {
    let c_in_per_group = c_in / group;
    let c_out_per_group = c_out / group;
    output.fill(0.0);

    for batch in 0..n {
        for g in 0..group {
            for oc in 0..c_out_per_group {
                let abs_oc = g * c_out_per_group + oc;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ic in 0..c_in_per_group {
                            let abs_ic = g * c_in_per_group + ic;
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let ih = oh * sh + fh * dh;
                                    let iw = ow * sw + fw * dw;
                                    if ih >= p0 && iw >= p1 && ih - p0 < h_in && iw - p1 < w_in {
                                        let ih = ih - p0;
                                        let iw = iw - p1;
                                        let input_idx =
                                            ((batch * c_in + abs_ic) * h_in + ih) * w_in + iw;
                                        let weight_idx =
                                            ((abs_oc * c_in_per_group + ic) * kh + fh) * kw + fw;
                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                        if let Some(bias) = bias {
                            sum += bias[abs_oc];
                        }
                        let out_idx = ((batch * c_out + abs_oc) * h_out + oh) * w_out + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }
}
