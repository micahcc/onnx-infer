use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::conv::AutoPad;

pub struct MaxPoolPrecomp {
    pub n: usize,
    pub c: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub p0: usize,
    pub p1: usize,
    pub total: usize,
}

pub struct MaxPool {
    pub inputs: Vec<String>,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pads: [usize; 4],
    pub auto_pad: AutoPad,
    pub precomp: Option<MaxPoolPrecomp>,
}

impl MaxPool {
    pub fn new(
        inputs: Vec<String>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        auto_pad: String,
        input_shape: &[usize],
    ) -> Result<Self> {
        if kernel_shape.is_empty() {
            return Err(InferenceError::InvalidModel(
                "MaxPool missing kernel_shape".into(),
            ));
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

        let precomp = if input_shape.len() == 4 {
            let ins = input_shape;
            let n = ins[0];
            let c = ins[1];
            let h_in = ins[2];
            let w_in = ins[3];
            let (p0, p1, p2, p3) = match auto_pad_enum {
                AutoPad::SameUpper | AutoPad::SameLower => {
                    let oh = h_in.div_ceil(sh);
                    let ow = w_in.div_ceil(sw);
                    let pad_h = ((oh - 1) * sh + kh).saturating_sub(h_in);
                    let pad_w = ((ow - 1) * sw + kw).saturating_sub(w_in);
                    match auto_pad_enum {
                        AutoPad::SameUpper => {
                            (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
                        }
                        _ => (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2),
                    }
                }
                AutoPad::Valid => (pads_arr[0], pads_arr[1], pads_arr[2], pads_arr[3]),
            };
            let h_out = (h_in + p0 + p2 - kh) / sh + 1;
            let w_out = (w_in + p1 + p3 - kw) / sw + 1;
            Some(MaxPoolPrecomp {
                n,
                c,
                h_in,
                w_in,
                h_out,
                w_out,
                p0,
                p1,
                total: n * c * h_out * w_out,
            })
        } else {
            None
        };

        Ok(Self {
            inputs,
            kh,
            kw,
            sh,
            sw,
            pads: pads_arr,
            auto_pad: auto_pad_enum,
            precomp,
        })
    }
}

impl Layer for MaxPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let (n, c, h_in, w_in, h_out, w_out, p0, p1, total) = if let Some(p) = &self.precomp
            && p.h_in == input.dims[2]
            && p.w_in == input.dims[3]
        {
            (
                p.n, p.c, p.h_in, p.w_in, p.h_out, p.w_out, p.p0, p.p1, p.total,
            )
        } else {
            let n = input.dims[0];
            let c = input.dims[1];
            let h_in = input.dims[2];
            let w_in = input.dims[3];
            let (p0, p1, p2, p3) = match self.auto_pad {
                AutoPad::SameUpper | AutoPad::SameLower => {
                    let oh = h_in.div_ceil(self.sh);
                    let ow = w_in.div_ceil(self.sw);
                    let pad_h = ((oh - 1) * self.sh + self.kh).saturating_sub(h_in);
                    let pad_w = ((ow - 1) * self.sw + self.kw).saturating_sub(w_in);
                    match self.auto_pad {
                        AutoPad::SameUpper => {
                            (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
                        }
                        _ => (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2),
                    }
                }
                AutoPad::Valid => (self.pads[0], self.pads[1], self.pads[2], self.pads[3]),
            };
            let h_out = (h_in + p0 + p2 - self.kh) / self.sh + 1;
            let w_out = (w_in + p1 + p3 - self.kw) / self.sw + 1;
            (
                n,
                c,
                h_in,
                w_in,
                h_out,
                w_out,
                p0,
                p1,
                n * c * h_out * w_out,
            )
        };

        let kh = self.kh;
        let kw = self.kw;
        let sh = self.sh;
        let sw = self.sw;
        let input_f = input.floats();
        let buf = output.as_mut_f32(total);
        buf.fill(f32::NEG_INFINITY);

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = oh * sh + fh;
                                let iw = ow * sw + fw;
                                if ih >= p0 && iw >= p1 && ih - p0 < h_in && iw - p1 < w_in {
                                    let ih = ih - p0;
                                    let iw = iw - p1;
                                    let idx = ((batch * c + ch) * h_in + ih) * w_in + iw;
                                    max_val = max_val.max(input_f[idx]);
                                }
                            }
                        }
                        let out_idx = ((batch * c + ch) * h_out + oh) * w_out + ow;
                        buf[out_idx] = max_val;
                    }
                }
            }
        }

        output.set_dims(&[n, c, h_out, w_out]);
        Ok(())
    }
}
