use anyhow::Context;
use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::conv::AutoPad;

pub struct AveragePoolPrecomp {
    pub n: usize,
    pub c: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
    pub p3: usize,
    pub total: usize,
}

pub struct AveragePool {
    pub inputs: Vec<String>,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pads: [usize; 4],
    pub auto_pad: AutoPad,
    pub count_include_pad: bool,
    pub shape_cache: Dims,
    pub precomp: Option<AveragePoolPrecomp>,
}

impl AveragePool {
    pub fn compute_shapes(
        shape: &[usize],
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        auto_pad: AutoPad,
        pads: &[usize; 4],
    ) -> AveragePoolPrecomp {
        let n = shape[0];
        let c = shape[1];
        let h_in = shape[2];
        let w_in = shape[3];
        let (p0, p1, p2, p3) = match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                let oh = h_in.div_ceil(sh);
                let ow = w_in.div_ceil(sw);
                let pad_h = ((oh - 1) * sh + kh).saturating_sub(h_in);
                let pad_w = ((ow - 1) * sw + kw).saturating_sub(w_in);
                match auto_pad {
                    AutoPad::SameUpper => {
                        (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
                    }
                    _ => (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2),
                }
            }
            AutoPad::Valid => (pads[0], pads[1], pads[2], pads[3]),
        };
        let h_out = (h_in + p0 + p2 - kh) / sh + 1;
        let w_out = (w_in + p1 + p3 - kw) / sw + 1;
        AveragePoolPrecomp {
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            p0,
            p1,
            p2,
            p3,
            total: n * c * h_out * w_out,
        }
    }

    pub fn new(
        inputs: Vec<String>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        auto_pad: String,
        count_include_pad: i64,
        initial_shape: &[usize],
    ) -> Result<Self> {
        if kernel_shape.is_empty() {
            anyhow::bail!("AveragePool missing kernel_shape");
        }
        let auto_pad_enum = match auto_pad.as_str() {
            "SAME_UPPER" => AutoPad::SameUpper,
            "SAME_LOWER" => AutoPad::SameLower,
            _ => AutoPad::Valid,
        };
        let kh = kernel_shape[0] as usize;
        let kw = kernel_shape[1] as usize;
        let sh = strides[0] as usize;
        let sw = strides[1] as usize;
        let pads_arr = [
            pads[0] as usize,
            pads[1] as usize,
            pads[2] as usize,
            pads[3] as usize,
        ];

        let (shape_cache, precomp) = if initial_shape.len() == 4 {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(
                    initial_shape,
                    kh,
                    kw,
                    sh,
                    sw,
                    auto_pad_enum,
                    &pads_arr,
                )),
            )
        } else {
            (Dims::new(), None)
        };

        Ok(Self {
            inputs,
            kh,
            kw,
            sh,
            sw,
            pads: pads_arr,
            auto_pad: auto_pad_enum,
            count_include_pad: count_include_pad != 0,
            shape_cache,
            precomp,
        })
    }
}

impl Layer for AveragePool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(
                    &input.dims,
                    self.kh,
                    self.kw,
                    self.sh,
                    self.sw,
                    self.auto_pad,
                    &self.pads,
                ));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let kh = self.kh;
        let kw = self.kw;
        let sh = self.sh;
        let sw = self.sw;
        let count_include_pad = self.count_include_pad;
        let kernel_area = kh * kw;
        let input_f = input.floats().context("in AveragePool layer")?;
        let buf = output.as_mut_f32(p.total);
        buf.fill(0.0);

        // Compute the range of output pixels that don't need bounds checks
        let oh_safe_start = if p.p0 > 0 { p.p0.div_ceil(sh) } else { 0 };
        let ow_safe_start = if p.p1 > 0 { p.p1.div_ceil(sw) } else { 0 };
        let oh_safe_end = if p.h_in + p.p0 >= kh {
            ((p.h_in + p.p0 - kh) / sh + 1).min(p.h_out)
        } else {
            0
        };
        let ow_safe_end = if p.w_in + p.p1 >= kw {
            ((p.w_in + p.p1 - kw) / sw + 1).min(p.w_out)
        } else {
            0
        };

        for batch in 0..p.n {
            for ch in 0..p.c {
                let in_base = (batch * p.c + ch) * p.h_in * p.w_in;
                let out_base = (batch * p.c + ch) * p.h_out * p.w_out;
                for oh in 0..p.h_out {
                    for ow in 0..p.w_out {
                        let mut sum = 0.0f32;
                        let mut valid_count = 0usize;
                        if oh >= oh_safe_start
                            && oh < oh_safe_end
                            && ow >= ow_safe_start
                            && ow < ow_safe_end
                        {
                            // No bounds checks needed — all kernel positions are valid
                            let ih_start = oh * sh - p.p0;
                            let iw_start = ow * sw - p.p1;
                            for fh in 0..kh {
                                let row = in_base + (ih_start + fh) * p.w_in + iw_start;
                                for fw in 0..kw {
                                    sum += input_f[row + fw];
                                }
                            }
                            valid_count = kernel_area;
                        } else {
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let ih = oh * sh + fh;
                                    let iw = ow * sw + fw;
                                    if ih >= p.p0
                                        && iw >= p.p1
                                        && ih - p.p0 < p.h_in
                                        && iw - p.p1 < p.w_in
                                    {
                                        let idx =
                                            in_base + (ih - p.p0) * p.w_in + (iw - p.p1);
                                        sum += input_f[idx];
                                        valid_count += 1;
                                    }
                                }
                            }
                        }
                        let denom = if count_include_pad {
                            kernel_area
                        } else {
                            valid_count
                        };
                        buf[out_base + oh * p.w_out + ow] =
                            if denom > 0 { sum / denom as f32 } else { 0.0 };
                    }
                }
            }
        }

        output.set_dims(&[p.n, p.c, p.h_out, p.w_out]);
        Ok(())
    }
}
