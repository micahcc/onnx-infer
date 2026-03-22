use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct ConvTranspose {
    pub inputs: Vec<String>,
    pub strides: [usize; 2],
    pub pads: [usize; 4], // [top, left, bottom, right]
    pub dilations: [usize; 2],
    pub group: usize,
}

impl ConvTranspose {
    pub fn new(
        inputs: Vec<String>,
        strides: &[i64],
        pads: &[i64],
        dilations: &[i64],
        group: i64,
    ) -> Self {
        Self {
            inputs,
            strides: [strides[0] as usize, strides[1] as usize],
            pads: [
                pads[0] as usize,
                pads[1] as usize,
                pads[2] as usize,
                pads[3] as usize,
            ],
            dilations: [dilations[0] as usize, dilations[1] as usize],
            group: group as usize,
        }
    }
}

impl Layer for ConvTranspose {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let weight = get_tensor(values, &self.inputs[1])?;
        let bias = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            Some(get_tensor(values, &self.inputs[2])?)
        } else {
            None
        };

        let inp = input.floats().context("ConvTranspose input")?;
        let w = weight.floats().context("ConvTranspose weight")?;

        let batch = input.dims[0];
        let c_in = input.dims[1];
        let ih = input.dims[2];
        let iw = input.dims[3];

        // Weight shape: [C_in, C_out/group, kH, kW]
        let c_out_per_group = weight.dims[1];
        let kh = weight.dims[2];
        let kw = weight.dims[3];
        let c_out = c_out_per_group * self.group;
        let c_in_per_group = c_in / self.group;

        let [sh, sw] = self.strides;
        let [pt, pl, pb, pr] = self.pads;
        let [dh, dw] = self.dilations;

        let oh = sh * (ih - 1) + dh * (kh - 1) + 1 - pt - pb;
        let ow = sw * (iw - 1) + dw * (kw - 1) + 1 - pl - pr;

        let out_numel = batch * c_out * oh * ow;
        let buf = output.as_mut_f32(out_numel);

        // Initialize with bias or zeros
        if let Some(bias_t) = &bias {
            let b = bias_t.floats().context("ConvTranspose bias")?;
            for n in 0..batch {
                for oc in 0..c_out {
                    let base = (n * c_out + oc) * oh * ow;
                    for i in 0..oh * ow {
                        buf[base + i] = b[oc];
                    }
                }
            }
        } else {
            buf.fill(0.0);
        }

        // ConvTranspose: for each input element, scatter it to the output
        // using the transposed convolution kernel
        for n in 0..batch {
            for g in 0..self.group {
                for ic in 0..c_in_per_group {
                    let abs_ic = g * c_in_per_group + ic;
                    for oc in 0..c_out_per_group {
                        let abs_oc = g * c_out_per_group + oc;
                        let w_base = (abs_ic * c_out_per_group + oc) * kh * kw;
                        let out_base = (n * c_out + abs_oc) * oh * ow;
                        let in_base = (n * c_in + abs_ic) * ih * iw;

                        for iy in 0..ih {
                            for ix in 0..iw {
                                let v = inp[in_base + iy * iw + ix];
                                if v == 0.0 {
                                    continue;
                                }
                                for ky in 0..kh {
                                    let oy_raw = iy * sh + ky * dh;
                                    if oy_raw < pt || oy_raw - pt >= oh {
                                        continue;
                                    }
                                    let oy = oy_raw - pt;
                                    for kx in 0..kw {
                                        let ox_raw = ix * sw + kx * dw;
                                        if ox_raw < pl || ox_raw - pl >= ow {
                                            continue;
                                        }
                                        let ox = ox_raw - pl;
                                        buf[out_base + oy * ow + ox] +=
                                            v * w[w_base + ky * kw + kx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        output.set_dims(&[batch, c_out, oh, ow]);
        Ok(())
    }
}
