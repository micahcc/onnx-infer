use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Where {
    pub inputs: Vec<String>,
}

impl Where {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Where {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let cond = get_tensor(values, &self.inputs[0])?;
        let x = get_tensor(values, &self.inputs[1])?;
        let y = get_tensor(values, &self.inputs[2])?;

        // Broadcast all three shapes together
        let mut tmp = [0usize; 8];
        let ndim_xy = x.dims.len().max(y.dims.len());
        broadcast_shape_into(&x.dims, &y.dims, &mut tmp[..ndim_xy]);
        let ndim = ndim_xy.max(cond.dims.len());
        let mut out_shape = [0usize; 8];
        broadcast_shape_into(&tmp[..ndim_xy], &cond.dims, &mut out_shape[..ndim]);
        let out_dims = &out_shape[..ndim];
        let numel: usize = out_dims.iter().product();

        let mut index = [0usize; 8];

        match x.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                let xf = x.floats().context("in Where layer")?;
                let yf = y.floats().context("in Where layer")?;
                for val in buf.iter_mut() {
                    let ci = broadcast_index(&index[..ndim], &cond.dims, out_dims);
                    let xi = broadcast_index(&index[..ndim], &x.dims, out_dims);
                    let yi = broadcast_index(&index[..ndim], &y.dims, out_dims);
                    let c = match cond.dtype() {
                        DType::Float => cond.floats().context("in Where layer")?[ci] != 0.0,
                        DType::Int64 => cond.ints().context("in Where layer")?[ci] != 0,
                        DType::String => unreachable!(),
                    };
                    *val = if c { xf[xi] } else { yf[yi] };

                    for d in (0..ndim).rev() {
                        index[d] += 1;
                        if index[d] < out_dims[d] {
                            break;
                        }
                        index[d] = 0;
                    }
                }
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(numel);
                let xi = x.ints().context("in Where layer")?;
                let yi = y.ints().context("in Where layer")?;
                for val in buf.iter_mut() {
                    let ci = broadcast_index(&index[..ndim], &cond.dims, out_dims);
                    let xidx = broadcast_index(&index[..ndim], &x.dims, out_dims);
                    let yidx = broadcast_index(&index[..ndim], &y.dims, out_dims);
                    let c = match cond.dtype() {
                        DType::Float => cond.floats().context("in Where layer")?[ci] != 0.0,
                        DType::Int64 => cond.ints().context("in Where layer")?[ci] != 0,
                        DType::String => unreachable!(),
                    };
                    *val = if c { xi[xidx] } else { yi[yidx] };

                    for d in (0..ndim).rev() {
                        index[d] += 1;
                        if index[d] < out_dims[d] {
                            break;
                        }
                        index[d] = 0;
                    }
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        output.set_dims(out_dims);
        Ok(())
    }
}
