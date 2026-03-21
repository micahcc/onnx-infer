use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

pub struct PRelu {
    pub inputs: Vec<String>,
}

impl PRelu {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for PRelu {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let slope = get_tensor(values, &self.inputs[1])?;
        let inp = input.floats().context("in PRelu layer")?;
        let slp = slope.floats().context("in PRelu layer")?;

        let ndim = input.dims.len().max(slope.dims.len());
        let mut out_shape = [0usize; 8];
        broadcast_shape_into(&input.dims, &slope.dims, &mut out_shape[..ndim]);
        let out_dims = &out_shape[..ndim];
        let numel: usize = out_dims.iter().product();
        let buf = output.as_mut_f32(numel);

        // Fast path: slope is scalar
        if slp.len() == 1 {
            let s = slp[0];
            for (o, &x) in buf.iter_mut().zip(inp.iter()) {
                *o = if x >= 0.0 { x } else { s * x };
            }
            output.set_dims(&input.dims);
            return Ok(());
        }

        // Fast path: identical shapes
        if input.dims.as_slice() == slope.dims.as_slice() {
            for (o, (&x, &s)) in buf.iter_mut().zip(inp.iter().zip(slp.iter())) {
                *o = if x >= 0.0 { x } else { s * x };
            }
            output.set_dims(out_dims);
            return Ok(());
        }

        // Fast path: per-channel broadcast where slope has one non-1 dim
        if inp.len() == numel && slp.len() > 1 && slp.len() < numel {
            let s_rank = slope.dims.len();
            let offset = ndim - s_rank;
            let mut inner = 1usize;
            let mut s_stride = 0usize;
            let mut single_axis = true;
            let mut non1_count = 0usize;
            for i in (0..s_rank).rev() {
                if slope.dims[i] != 1 {
                    non1_count += 1;
                    if non1_count == 1 {
                        s_stride = inner;
                    } else {
                        single_axis = false;
                        break;
                    }
                }
                inner *= out_dims[i + offset];
            }
            if single_axis && non1_count == 1 {
                let s_len = slp.len();
                for i in 0..numel {
                    let si = (i / s_stride) % s_len;
                    let x = inp[i];
                    buf[i] = if x >= 0.0 { x } else { slp[si] * x };
                }
                output.set_dims(out_dims);
                return Ok(());
            }
        }

        // General fallback with per-element broadcast index
        let mut index = [0usize; 8];
        for val in buf.iter_mut() {
            let xi = broadcast_index(&index[..ndim], &input.dims, out_dims);
            let si = broadcast_index(&index[..ndim], &slope.dims, out_dims);
            let x = inp[xi];
            *val = if x >= 0.0 { x } else { slp[si] * x };

            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < out_dims[d] {
                    break;
                }
                index[d] = 0;
            }
        }

        output.set_dims(out_dims);
        Ok(())
    }
}
