use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Layout;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::conv::AutoPad;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct MaxPool {
    pub inputs: Vec<String>,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pads: [usize; 4],
    pub auto_pad: AutoPad,
    pub nhwc: bool,
    pub shape_cache: Dims,
    pub precomp: Option<MaxPoolPrecomp>,
}

impl MaxPool {
    pub fn compute_shapes(
        shape: &[usize],
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        auto_pad: AutoPad,
        pads: &[usize; 4],
        nhwc: bool,
    ) -> MaxPoolPrecomp {
        assert!(nhwc, "MaxPool::compute_shapes requires NHWC");
        let (n, c, h_in, w_in) = (shape[0], shape[3], shape[1], shape[2]);
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
        MaxPoolPrecomp {
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            p0,
            p1,
            total: n * c * h_out * w_out,
        }
    }

    pub fn new(
        inputs: Vec<String>,
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        auto_pad: String,
        initial_shape: &[usize],
        nhwc: bool,
    ) -> Result<Self> {
        if kernel_shape.is_empty() {
            anyhow::bail!("MaxPool missing kernel_shape");
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
                    nhwc,
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
            nhwc,
            shape_cache,
            precomp,
        })
    }
}

impl Layer for MaxPool {
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
        let input_f = input.floats().context("in MaxPool layer")?;
        let buf = output.as_mut_f32(p.total);
        buf.fill(f32::NEG_INFINITY);

        assert!(self.nhwc, "MaxPool::execute requires NHWC input layout");

        // NHWC: input[batch][h][w][c], output[batch][h_out][w_out][c]
        for batch in 0..p.n {
            let in_batch = batch * p.h_in * p.w_in * p.c;
            let out_batch = batch * p.h_out * p.w_out * p.c;
            for oh in 0..p.h_out {
                for ow in 0..p.w_out {
                    let out_off = out_batch + (oh * p.w_out + ow) * p.c;
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * sh + fh;
                            let iw = ow * sw + fw;
                            if ih >= p.p0 && iw >= p.p1 && ih - p.p0 < p.h_in && iw - p.p1 < p.w_in
                            {
                                let in_off = in_batch + ((ih - p.p0) * p.w_in + (iw - p.p1)) * p.c;
                                for ch in 0..p.c {
                                    buf[out_off + ch] = buf[out_off + ch].max(input_f[in_off + ch]);
                                }
                            }
                        }
                    }
                }
            }
        }
        output.set_dims(&[p.n, p.h_out, p.w_out, p.c]);
        output.layout = Layout::NHWC;

        Ok(())
    }
}
