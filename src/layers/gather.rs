use std::collections::HashMap;

use crate::DType;
use crate::InferenceError;
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

        if indices.dims.is_empty()
            || (indices.dims.len() == 1 && indices.dims[0] == 1)
            || indices.numel() == 1
        {
            let idx_i64 = indices.i64_at(0);
            let idx = if idx_i64 < 0 {
                (input.dims[axis] as i64 + idx_i64) as usize
            } else {
                idx_i64 as usize
            };

            if input.dims.len() == 1 {
                match input.dtype {
                    DType::Float => {
                        let buf = output.as_mut_f32(1);
                        buf[0] = input.floats()[idx];
                    }
                    DType::Int64 => {
                        let buf = output.as_mut_i64(1);
                        buf[0] = input.ints()[idx];
                    }
                }
                output.dims = vec![];
            } else {
                let outer: usize = input.dims[..axis].iter().product();
                let inner: usize = input.dims[axis + 1..].iter().product();
                let axis_size = input.dims[axis];
                let mut out_dims: Vec<usize> = input.dims[..axis].to_vec();
                out_dims.extend_from_slice(&input.dims[axis + 1..]);

                match input.dtype {
                    DType::Float => {
                        let d = input.floats();
                        let buf = output.as_mut_f32(outer * inner);
                        for o in 0..outer {
                            let base = o * axis_size * inner + idx * inner;
                            let dst = o * inner;
                            buf[dst..dst + inner].copy_from_slice(&d[base..base + inner]);
                        }
                    }
                    DType::Int64 => {
                        let d = input.ints();
                        let buf = output.as_mut_i64(outer * inner);
                        for o in 0..outer {
                            let base = o * axis_size * inner + idx * inner;
                            let dst = o * inner;
                            buf[dst..dst + inner].copy_from_slice(&d[base..base + inner]);
                        }
                    }
                }
                output.dims = out_dims;
            }
        } else {
            return Err(InferenceError::UnsupportedOperator(
                "Gather with non-scalar indices".into(),
            ));
        }
        Ok(())
    }
}
