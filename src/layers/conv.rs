use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Conv {
    pub inputs: Vec<String>,
    pub kernel_shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub pads: Vec<i64>,
    pub dilations: Vec<i64>,
    pub group: usize,
    pub auto_pad: String,
}

impl Conv {
    pub fn new(
        inputs: Vec<String>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: usize,
        auto_pad: String,
    ) -> Self {
        Self {
            inputs,
            kernel_shape,
            strides,
            pads,
            dilations,
            group,
            auto_pad,
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

        let n = input.dims[0];
        let c_in = input.dims[1];
        let h_in = input.dims[2];
        let w_in = input.dims[3];

        let c_out = weight.dims[0];
        let kh = if self.kernel_shape.is_empty() { weight.dims[2] } else { self.kernel_shape[0] as usize };
        let kw = if self.kernel_shape.is_empty() { weight.dims[3] } else { self.kernel_shape[1] as usize };
        let sh = self.strides[0] as usize;
        let sw = self.strides[1] as usize;
        let dh = self.dilations[0] as usize;
        let dw = self.dilations[1] as usize;

        let (p0, p1, p2, p3) = if self.auto_pad == "SAME_UPPER" || self.auto_pad == "SAME_LOWER" {
            let oh = h_in.div_ceil(sh);
            let ow = w_in.div_ceil(sw);
            let pad_h = ((oh - 1) * sh + dh * (kh - 1) + 1).saturating_sub(h_in);
            let pad_w = ((ow - 1) * sw + dw * (kw - 1) + 1).saturating_sub(w_in);
            if self.auto_pad == "SAME_UPPER" {
                (pad_h / 2, pad_w / 2, pad_h - pad_h / 2, pad_w - pad_w / 2)
            } else {
                (pad_h - pad_h / 2, pad_w - pad_w / 2, pad_h / 2, pad_w / 2)
            }
        } else {
            (self.pads[0] as usize, self.pads[1] as usize, self.pads[2] as usize, self.pads[3] as usize)
        };

        let h_out = (h_in + p0 + p2 - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w_in + p1 + p3 - dw * (kw - 1) - 1) / sw + 1;

        let c_in_per_group = c_in / self.group;
        let c_out_per_group = c_out / self.group;

        let input_f = input.floats();
        let weight_f = weight.floats();
        let total = n * c_out * h_out * w_out;
        let buf = output.as_mut_f32(total);
        buf.fill(0.0);

        for batch in 0..n {
            for g in 0..self.group {
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
                                        if ih >= p0
                                            && iw >= p1
                                            && ih - p0 < h_in
                                            && iw - p1 < w_in
                                        {
                                            let ih = ih - p0;
                                            let iw = iw - p1;
                                            let input_idx =
                                                ((batch * c_in + abs_ic) * h_in + ih) * w_in + iw;
                                            let weight_idx =
                                                ((abs_oc * c_in_per_group + ic) * kh + fh) * kw
                                                    + fw;
                                            sum += input_f[input_idx] * weight_f[weight_idx];
                                        }
                                    }
                                }
                            }
                            if let Some(ref bias) = bias {
                                sum += bias.floats()[abs_oc];
                            }
                            let out_idx = ((batch * c_out + abs_oc) * h_out + oh) * w_out + ow;
                            buf[out_idx] = sum;
                        }
                    }
                }
            }
        }

        output.set_dims(&[n, c_out, h_out, w_out]);
        Ok(())
    }
}
