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
        }
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

        let sh = self.sh;
        let sw = self.sw;
        let dh = self.dh;
        let dw = self.dw;
        let kh = self.kh;
        let kw = self.kw;

        let input_f = input.floats();
        let weight_f = weight.floats();
        let buf = output.as_mut_f32(p.total);
        buf.fill(0.0);

        for batch in 0..p.n {
            for g in 0..self.group {
                for oc in 0..p.c_out_per_group {
                    let abs_oc = g * p.c_out_per_group + oc;
                    for oh in 0..p.h_out {
                        for ow in 0..p.w_out {
                            let mut sum = 0.0f32;
                            for ic in 0..p.c_in_per_group {
                                let abs_ic = g * p.c_in_per_group + ic;
                                for fh in 0..kh {
                                    for fw in 0..kw {
                                        let ih = oh * sh + fh * dh;
                                        let iw = ow * sw + fw * dw;
                                        if ih >= p.p0
                                            && iw >= p.p1
                                            && ih - p.p0 < p.h_in
                                            && iw - p.p1 < p.w_in
                                        {
                                            let ih = ih - p.p0;
                                            let iw = iw - p.p1;
                                            let input_idx =
                                                ((batch * p.c_in + abs_ic) * p.h_in + ih) * p.w_in
                                                    + iw;
                                            let weight_idx =
                                                ((abs_oc * p.c_in_per_group + ic) * kh + fh) * kw
                                                    + fw;
                                            sum += input_f[input_idx] * weight_f[weight_idx];
                                        }
                                    }
                                }
                            }
                            if let Some(bias) = bias {
                                sum += bias.floats()[abs_oc];
                            }
                            let out_idx =
                                ((batch * p.c_out + abs_oc) * p.h_out + oh) * p.w_out + ow;
                            buf[out_idx] = sum;
                        }
                    }
                }
            }
        }

        output.set_dims(&[p.n, p.c_out, p.h_out, p.w_out]);
        Ok(())
    }
}
