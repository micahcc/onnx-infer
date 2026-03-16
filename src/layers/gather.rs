use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Gather {
    pub inputs: Vec<String>,
    pub axis: i64,
}

impl Gather {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for Gather {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let indices = get_tensor(values, &self.inputs[1])?;
        let rank = input.dims.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        let outer: usize = input.dims[..axis].iter().product();
        let axis_size = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();
        let num_indices = indices.numel();

        // Output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
        let mut out_dims = [0usize; 8];
        let mut out_rank = 0;
        for &d in &input.dims[..axis] {
            out_dims[out_rank] = d;
            out_rank += 1;
        }
        for &d in &indices.dims {
            out_dims[out_rank] = d;
            out_rank += 1;
        }
        for &d in &input.dims[axis + 1..] {
            out_dims[out_rank] = d;
            out_rank += 1;
        }

        let numel = outer * num_indices * inner;

        // Resolve indices (handle negatives)
        let idx_vals: &[i64];
        let idx_converted: Vec<i64>;
        match indices.dtype() {
            DType::Int64 => idx_vals = indices.ints(),
            DType::Float => {
                idx_converted = indices.floats().iter().map(|&v| v as i64).collect();
                idx_vals = &idx_converted;
            }
        }

        match input.dtype() {
            DType::Float => {
                let d = input.floats();
                let buf = output.as_mut_f32(numel);
                let mut dst = 0;
                for o in 0..outer {
                    for &raw_idx in idx_vals {
                        let idx = if raw_idx < 0 {
                            (axis_size as i64 + raw_idx) as usize
                        } else {
                            raw_idx as usize
                        };
                        let base = o * axis_size * inner + idx * inner;
                        buf[dst..dst + inner].copy_from_slice(&d[base..base + inner]);
                        dst += inner;
                    }
                }
            }
            DType::Int64 => {
                let d = input.ints();
                let buf = output.as_mut_i64(numel);
                let mut dst = 0;
                for o in 0..outer {
                    for &raw_idx in idx_vals {
                        let idx = if raw_idx < 0 {
                            (axis_size as i64 + raw_idx) as usize
                        } else {
                            raw_idx as usize
                        };
                        let base = o * axis_size * inner + idx * inner;
                        buf[dst..dst + inner].copy_from_slice(&d[base..base + inner]);
                        dst += inner;
                    }
                }
            }
        }
        output.set_dims(&out_dims[..out_rank]);
        Ok(())
    }
}
