use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Compress {
    pub inputs: Vec<String>,
    pub axis: Option<i64>,
}

impl Compress {
    pub fn new(inputs: Vec<String>, axis: Option<i64>) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for Compress {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let condition = get_tensor(values, &self.inputs[1])?;

        // Get boolean condition indices
        let cond_true: Vec<usize> = match condition.dtype() {
            DType::Int64 => condition
                .ints()
                .context("in Compress layer")?
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0)
                .map(|(i, _)| i)
                .collect(),
            DType::String => unreachable!("strings not supported"),
            DType::Float => condition
                .floats()
                .context("in Compress layer")?
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(i, _)| i)
                .collect(),
        };

        match self.axis {
            None => {
                // Flatten input, select elements by condition
                let count = cond_true.len();
                match input.dtype() {
                    DType::Float => {
                        let data = input.floats().context("in Compress layer")?;
                        let buf = output.as_mut_f32(count);
                        for (out_i, &in_i) in cond_true.iter().enumerate() {
                            buf[out_i] = data[in_i];
                        }
                    }
                    DType::Int64 => {
                        let data = input.ints().context("in Compress layer")?;
                        let buf = output.as_mut_i64(count);
                        for (out_i, &in_i) in cond_true.iter().enumerate() {
                            buf[out_i] = data[in_i];
                        }
                    }
                    DType::String => unreachable!("strings not supported"),
                }
                output.set_dims(&[count]);
            }
            Some(axis_val) => {
                let rank = input.dims.len();
                let axis = if axis_val < 0 {
                    (rank as i64 + axis_val) as usize
                } else {
                    axis_val as usize
                };

                let count = cond_true.len();
                let mut out_dims: Vec<usize> = input.dims.to_vec();
                out_dims[axis] = count;
                let out_numel: usize = out_dims.iter().product();

                let outer: usize = input.dims[..axis].iter().product();
                let axis_len = input.dims[axis];
                let inner: usize = input.dims[axis + 1..].iter().product();

                match input.dtype() {
                    DType::Float => {
                        let data = input.floats().context("in Compress layer")?;
                        let buf = output.as_mut_f32(out_numel);
                        for o in 0..outer {
                            for (new_a, &old_a) in cond_true.iter().enumerate() {
                                for i in 0..inner {
                                    let src = o * axis_len * inner + old_a * inner + i;
                                    let dst = o * count * inner + new_a * inner + i;
                                    buf[dst] = data[src];
                                }
                            }
                        }
                    }
                    DType::Int64 => {
                        let data = input.ints().context("in Compress layer")?;
                        let buf = output.as_mut_i64(out_numel);
                        for o in 0..outer {
                            for (new_a, &old_a) in cond_true.iter().enumerate() {
                                for i in 0..inner {
                                    let src = o * axis_len * inner + old_a * inner + i;
                                    let dst = o * count * inner + new_a * inner + i;
                                    buf[dst] = data[src];
                                }
                            }
                        }
                    }
                    DType::String => unreachable!("strings not supported"),
                }
                output.set_dims(&out_dims);
            }
        }
        Ok(())
    }
}
