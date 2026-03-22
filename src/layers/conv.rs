use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Layout;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug, Clone, Copy)]
pub enum AutoPad {
    Valid,
    SameUpper,
    SameLower,
}

#[derive(Debug)]
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

#[derive(Debug)]
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
    pub nhwc: bool,
    pub shape_cache: Dims,
    pub precomp: Option<ConvPrecomp>,
    col_buf: Vec<f32>,
    gemm_buf: Vec<f32>,
}

impl Conv {
    /// Extract spatial dimensions from input shape based on layout.
    /// Returns (n, c_in, h_in, w_in) regardless of layout.
    fn extract_dims(ins: &[usize], nhwc: bool) -> (usize, usize, usize, usize) {
        if nhwc {
            (ins[0], ins[3], ins[1], ins[2])
        } else {
            (ins[0], ins[1], ins[2], ins[3])
        }
    }

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
        nhwc: bool,
    ) -> ConvPrecomp {
        let (n, c_in, h_in, w_in) = Self::extract_dims(ins, nhwc);
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
        nhwc: bool,
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
                    nhwc,
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
            nhwc,
            shape_cache,
            precomp,
            col_buf: Vec::new(),
            gemm_buf: Vec::new(),
        }
    }
}

/// im2col for NHWC input layout.
///
/// Input slice starts at the batch offset: `&input[batch * H * W * C ..]`
/// Memory order: `input[ih * W * C + iw * C + channel]`
///
/// `group_ch_offset` is `g * c_in_per_group` — the channel offset for the group.
fn im2col_nhwc(
    input: &[f32],
    col: &mut [f32],
    c_in: usize,
    c_in_per_group: usize,
    group_ch_offset: usize,
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
                            // NHWC: input[ih * W * C + iw * C + channel]
                            col[col_idx] = input
                                [(ih_actual * w_in + (iw - p1)) * c_in + group_ch_offset + ic];
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
    /// Always operates in NCHW layout.
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
                    false, // naive always NCHW
                ));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let input_f = input.floats().context("in Conv layer")?;
        let weight_f = weight.floats().context("in Conv layer")?;
        let buf = output.as_mut_f32(p.total);
        conv_naive(
            input_f,
            weight_f,
            bias.map(|b| b.floats()).transpose()?,
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
                    self.nhwc,
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

        let input_f = input.floats().context("in Conv layer")?;
        let weight_f = weight.floats().context("in Conv layer")?;
        let buf = output.as_mut_f32(p.total);

        let col_size = p.c_in_per_group * kh * kw * spatial_out;
        self.col_buf.resize(col_size, 0.0);

        let gemm_m = p.c_out_per_group;
        let gemm_k = p.c_in_per_group * kh * kw;
        let gemm_n = spatial_out;

        assert!(self.nhwc, "Conv::execute requires NHWC input layout");

        // NHWC path: im2col from NHWC input, GEMM to temp buffer, scatter to NHWC output
        let group_gemm_size = p.c_out_per_group * spatial_out;
        self.gemm_buf.resize(group_gemm_size, 0.0);

        for batch in 0..p.n {
            let in_batch = batch * p.h_in * p.w_in * p.c_in;

            for g in 0..self.group {
                im2col_nhwc(
                    &input_f[in_batch..],
                    &mut self.col_buf,
                    p.c_in,
                    p.c_in_per_group,
                    g * p.c_in_per_group,
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

                // Fill gemm_buf with bias if present
                if let Some(bias) = bias {
                    let bias_f = bias.floats().context("in Conv layer")?;
                    for oc in 0..gemm_m {
                        let abs_oc = g * p.c_out_per_group + oc;
                        let row_start = oc * spatial_out;
                        for s in 0..spatial_out {
                            self.gemm_buf[row_start + s] = bias_f[abs_oc];
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
                        &mut self.gemm_buf,
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
                        &mut self.gemm_buf,
                        gemm_n,
                    );
                }

                // Scatter GEMM result [c_out_per_group, spatial_out] to NHWC output
                // NHWC output: buf[(batch * H_out * W_out + s) * C_out + abs_oc]
                let out_batch = batch * p.h_out * p.w_out * p.c_out;
                for oc in 0..p.c_out_per_group {
                    let abs_oc = g * p.c_out_per_group + oc;
                    for s in 0..spatial_out {
                        buf[out_batch + s * p.c_out + abs_oc] =
                            self.gemm_buf[oc * spatial_out + s];
                    }
                }
            }
        }

        output.set_dims(&[p.n, p.h_out, p.w_out, p.c_out]);
        output.layout = Layout::NHWC;

        Ok(())
    }
}

/// Naive reference implementation for correctness testing (NCHW only).
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
