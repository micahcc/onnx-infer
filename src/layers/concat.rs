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
    // Precomputed (0 = not precomputed)
    pub pre_axis: usize,
    pub pre_rank: usize,
    pub pre_outer: usize,
    pub pre_inner: usize,
}

impl Concat {
    pub fn new(inputs: Vec<String>, axis: i64, input_shapes: &[&[usize]]) -> Self {
        let mut s = Self {
            inputs,
            axis,
            pre_axis: 0,
            pre_rank: 0,
            pre_outer: 0,
            pre_inner: 0,
        };
        // Use the first non-empty input shape to precompute
        if let Some(shape) = input_shapes.iter().find(|s| !s.is_empty()) {
            let rank = shape.len();
            let axis = if axis < 0 {
                (rank as i64 + axis) as usize
            } else {
                axis as usize
            };
            s.pre_axis = axis;
            s.pre_rank = rank;
            s.pre_outer = shape[..axis].iter().product();
            s.pre_inner = shape[axis + 1..].iter().product();
        }
        s
    }
}

impl Layer for Concat {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let first_name = self
            .inputs
            .iter()
            .find(|n| !n.is_empty())
            .ok_or_else(|| InferenceError::InvalidModel("Concat with no inputs".into()))?;
        let first = get_tensor(values, first_name)?;

        let (axis, rank, outer, inner) = if self.pre_rank > 0 {
            (self.pre_axis, self.pre_rank, self.pre_outer, self.pre_inner)
        } else {
            let rank = first.dims.len();
            let axis = if self.axis < 0 {
                (rank as i64 + self.axis) as usize
            } else {
                self.axis as usize
            };
            let outer: usize = first.dims[..axis].iter().product();
            let inner: usize = first.dims[axis + 1..].iter().product();
            (axis, rank, outer, inner)
        };

        let mut out_dims = [0usize; 8];
        for (i, &d) in first.dims.iter().enumerate() {
            out_dims[i] = d;
        }
        let mut axis_sum = 0usize;
        for name in &self.inputs {
            if !name.is_empty() {
                let t = get_tensor(values, name)?;
                axis_sum += t.dims[axis];
            }
        }
        out_dims[axis] = axis_sum;

        let dtype = first.dtype();

        match dtype {
            DType::Int64 => {
                let total = out_dims[..rank].iter().product::<usize>();
                let buf = output.as_mut_i64(total);
                let mut axis_offset = 0;
                for name in &self.inputs {
                    if name.is_empty() {
                        continue;
                    }
                    let t = get_tensor(values, name)?;
                    let t_axis = t.dims[axis];
                    if t.dtype() == DType::Float {
                        let floats = t.floats();
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                for k in 0..inner {
                                    buf[dst_base + k] = floats[src_base + k] as i64;
                                }
                            }
                        }
                    } else {
                        let t_data = t.ints();
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                buf[dst_base..dst_base + inner]
                                    .copy_from_slice(&t_data[src_base..src_base + inner]);
                            }
                        }
                    }
                    axis_offset += t_axis;
                }
            }
            DType::Float => {
                let total = out_dims[..rank].iter().product::<usize>();
                let buf = output.as_mut_f32(total);
                let mut axis_offset = 0;
                for name in &self.inputs {
                    if name.is_empty() {
                        continue;
                    }
                    let t = get_tensor(values, name)?;
                    let t_axis = t.dims[axis];
                    if t.dtype() == DType::Int64 {
                        let ints = t.ints();
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                for k in 0..inner {
                                    buf[dst_base + k] = ints[src_base + k] as f32;
                                }
                            }
                        }
                    } else {
                        let t_data = t.floats();
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                buf[dst_base..dst_base + inner]
                                    .copy_from_slice(&t_data[src_base..src_base + inner]);
                            }
                        }
                    }
                    axis_offset += t_axis;
                }
            }
        }
        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
