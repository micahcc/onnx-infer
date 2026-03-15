use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct MaxPool {
    pub inputs: Vec<String>,
    pub kernel_shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub pads: Vec<i64>,
    pub auto_pad: String,
}

impl MaxPool {
    pub fn new(
        inputs: Vec<String>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        auto_pad: String,
    ) -> Result<Self> {
        if kernel_shape.is_empty() {
            return Err(InferenceError::InvalidModel(
                "MaxPool missing kernel_shape".into(),
            ));
        }
        Ok(Self {
            inputs,
            kernel_shape,
            strides,
            pads,
            auto_pad,
        })
    }
}

impl Layer for MaxPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let n = input.dims[0];
        let c = input.dims[1];
        let h_in = input.dims[2];
        let w_in = input.dims[3];

        let kh = self.kernel_shape[0] as usize;
        let kw = self.kernel_shape[1] as usize;
        let sh = self.strides[0] as usize;
        let sw = self.strides[1] as usize;

        let mut pads = self.pads.clone();
        if self.auto_pad == "SAME_UPPER" || self.auto_pad == "SAME_LOWER" {
            let oh = h_in.div_ceil(sh);
            let ow = w_in.div_ceil(sw);
            let pad_h = ((oh - 1) * sh + kh).saturating_sub(h_in);
            let pad_w = ((ow - 1) * sw + kw).saturating_sub(w_in);
            if self.auto_pad == "SAME_UPPER" {
                pads = vec![
                    (pad_h / 2) as i64,
                    (pad_w / 2) as i64,
                    (pad_h - pad_h / 2) as i64,
                    (pad_w - pad_w / 2) as i64,
                ];
            } else {
                pads = vec![
                    (pad_h - pad_h / 2) as i64,
                    (pad_w - pad_w / 2) as i64,
                    (pad_h / 2) as i64,
                    (pad_w / 2) as i64,
                ];
            }
        }

        let ph_begin = pads[0] as usize;
        let pw_begin = pads[1] as usize;

        let h_out = (h_in + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
        let w_out = (w_in + pads[1] as usize + pads[3] as usize - kw) / sw + 1;

        let input_f = input.floats();
        let total = n * c * h_out * w_out;
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
                                if ih >= ph_begin
                                    && iw >= pw_begin
                                    && ih - ph_begin < h_in
                                    && iw - pw_begin < w_in
                                {
                                    let ih = ih - ph_begin;
                                    let iw = iw - pw_begin;
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

        output.dims = vec![n, c, h_out, w_out];
        Ok(())
    }
}
