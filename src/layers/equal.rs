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
pub struct Equal {
    pub inputs: Vec<String>,
}

impl Equal {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Equal {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let ndim = a.dims.len().max(b.dims.len());
        let mut out_shape = [0usize; 8];
        broadcast_shape_into(&a.dims, &b.dims, &mut out_shape[..ndim]);
        let out_dims = &out_shape[..ndim];
        let numel: usize = out_dims.iter().product();

        if numel == 0 {
            output.as_mut_i64(0);
            output.set_dims(out_dims);
            return Ok(());
        }

        let buf = output.as_mut_i64(numel);
        let mut index = [0usize; 8];

        if a.dtype() == DType::Int64 && b.dtype() == DType::Int64 {
            let ad = a.ints().context("in Equal layer")?;
            let bd = b.ints().context("in Equal layer")?;
            for val in buf.iter_mut() {
                let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
                let bi = broadcast_index(&index[..ndim], &b.dims, out_dims);
                *val = if ad[ai] == bd[bi] { 1 } else { 0 };
                for d in (0..ndim).rev() {
                    index[d] += 1;
                    if index[d] < out_dims[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
        } else if a.dtype() == DType::Float && b.dtype() == DType::Float {
            let ad = a.floats().context("in Equal layer")?;
            let bd = b.floats().context("in Equal layer")?;
            for val in buf.iter_mut() {
                let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
                let bi = broadcast_index(&index[..ndim], &b.dims, out_dims);
                *val = if ad[ai] == bd[bi] { 1 } else { 0 };
                for d in (0..ndim).rev() {
                    index[d] += 1;
                    if index[d] < out_dims[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
        } else {
            // Mixed types: compare as f64 to avoid precision loss
            for val in buf.iter_mut() {
                let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
                let bi = broadcast_index(&index[..ndim], &b.dims, out_dims);
                let va: f64 = match a.dtype() {
                    DType::Float => a.floats().context("in Equal layer")?[ai] as f64,
                    DType::Int64 => a.ints().context("in Equal layer")?[ai] as f64,
                    DType::String => unreachable!("strings not supported"),
                };
                let vb: f64 = match b.dtype() {
                    DType::Float => b.floats().context("in Equal layer")?[bi] as f64,
                    DType::Int64 => b.ints().context("in Equal layer")?[bi] as f64,
                    DType::String => unreachable!("strings not supported"),
                };
                *val = if va == vb { 1 } else { 0 };
                for d in (0..ndim).rev() {
                    index[d] += 1;
                    if index[d] < out_dims[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
        }

        output.set_dims(out_dims);
        Ok(())
    }
}
