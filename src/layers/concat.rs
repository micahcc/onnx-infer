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
        let first_name = self.inputs.iter().find(|n| !n.is_empty())
            .ok_or_else(|| InferenceError::InvalidModel("Concat with no inputs".into()))?;
        let first = get_tensor(values, first_name)?;

        let rank = first.dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let mut out_dims = [0usize; 8];
        for (i, &d) in first.dims.iter().enumerate() {
            out_dims[i] = d;
        }
        // Sum the axis dimension across all inputs
        let mut axis_sum = 0usize;
        for name in &self.inputs {
            if !name.is_empty() {
                let t = get_tensor(values, name)?;
                axis_sum += t.dims[axis];
            }
        }
        out_dims[axis] = axis_sum;

        let outer: usize = out_dims[..axis].iter().product();
        let inner: usize = out_dims[axis + 1..rank].iter().product();

        let is_int = first.dtype() == DType::Int64;
        if is_int {
            let total = out_dims[..rank].iter().product::<usize>();
            let buf = output.as_mut_i64(total);
            let mut axis_offset = 0;
            for name in &self.inputs {
                if name.is_empty() { continue; }
                let t = get_tensor(values, name)?;
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
            let total = out_dims[..rank].iter().product::<usize>();
            let buf = output.as_mut_f32(total);
            let mut axis_offset = 0;
            for name in &self.inputs {
                if name.is_empty() { continue; }
                let t = get_tensor(values, name)?;
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
        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
