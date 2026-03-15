use std::collections::HashMap;

use crate::DType;
use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Concat {
    pub inputs: Vec<String>,
    pub axis: i64,
}

impl Concat {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for Concat {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let tensors: Vec<&Tensor> = self
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| get_tensor(values, name))
            .collect::<Result<Vec<_>>>()?;

        if tensors.is_empty() {
            return Err(InferenceError::InvalidModel("Concat with no inputs".into()));
        }

        let rank = tensors[0].dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let mut out_dims = tensors[0].dims.clone();
        out_dims[axis] = tensors.iter().map(|t| t.dims[axis]).sum();

        let outer: usize = out_dims[..axis].iter().product();
        let inner: usize = out_dims[axis + 1..].iter().product();

        let is_int = tensors[0].dtype == DType::Int64;
        if is_int {
            let total = out_dims.iter().product::<usize>();
            let buf = output.as_mut_i64(total);
            let mut axis_offset = 0;
            for t in &tensors {
                let t_data = t.ints();
                let t_axis = t.dims[axis];
                for o in 0..outer {
                    for a in 0..t_axis {
                        let src_base = (o * t_axis + a) * inner;
                        let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                        buf[dst_base..dst_base + inner]
                            .copy_from_slice(&t_data[src_base..src_base + inner]);
                    }
                }
                axis_offset += t_axis;
            }
        } else {
            let total = out_dims.iter().product::<usize>();
            let buf = output.as_mut_f32(total);
            let mut axis_offset = 0;
            for t in &tensors {
                let t_data = t.floats();
                let t_axis = t.dims[axis];
                for o in 0..outer {
                    for a in 0..t_axis {
                        let src_base = (o * t_axis + a) * inner;
                        let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                        buf[dst_base..dst_base + inner]
                            .copy_from_slice(&t_data[src_base..src_base + inner]);
                    }
                }
                axis_offset += t_axis;
            }
        }
        output.dims = out_dims;
        Ok(())
    }
}
