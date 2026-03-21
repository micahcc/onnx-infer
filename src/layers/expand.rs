use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Expand {
    pub inputs: Vec<String>,
}

impl Expand {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Expand {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let shape_tensor = get_tensor(values, &self.inputs[1])?;

        let mut target = [0usize; 8];
        let target_len = shape_tensor.numel();
        match shape_tensor.dtype() {
            DType::Int64 => {
                for (i, &v) in shape_tensor
                    .ints()
                    .context("in Expand layer")?
                    .iter()
                    .enumerate()
                {
                    target[i] = v as usize;
                }
            }
            DType::Float => {
                for (i, &v) in shape_tensor
                    .floats()
                    .context("in Expand layer")?
                    .iter()
                    .enumerate()
                {
                    target[i] = v as usize;
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        let ndim = input.dims.len().max(target_len);
        let mut out_shape = [0usize; 8];
        broadcast_shape_into(&input.dims, &target[..target_len], &mut out_shape[..ndim]);
        let out_dims = &out_shape[..ndim];
        let numel: usize = out_dims.iter().product();

        let mut index = [0usize; 8];
        match input.dtype() {
            DType::Float => {
                let in_data = input.floats().context("in Expand layer")?;
                let buf = output.as_mut_f32(numel);
                for val in buf.iter_mut() {
                    let ai = broadcast_index(&index[..ndim], &input.dims, out_dims);
                    *val = in_data[ai];
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
                let in_data = input.ints().context("in Expand layer")?;
                let buf = output.as_mut_i64(numel);
                for val in buf.iter_mut() {
                    let ai = broadcast_index(&index[..ndim], &input.dims, out_dims);
                    *val = in_data[ai];
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
