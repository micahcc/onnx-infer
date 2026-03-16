use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct GatherPrecomp {
    pub axis: usize,
    pub outer: usize,
    pub axis_size: usize,
    pub inner: usize,
    pub out_dims: [usize; 8],
    pub out_rank: usize,
}

pub struct Gather {
    pub inputs: Vec<String>,
    pub axis: i64,
    pub precomp: Option<GatherPrecomp>,
}

impl Gather {
    pub fn new(
        inputs: Vec<String>,
        axis: i64,
        data_shape: &[usize],
        indices_shape: &[usize],
    ) -> Self {
        let precomp = if !data_shape.is_empty() && !indices_shape.is_empty() {
            let (ds, is) = (data_shape, indices_shape);
            let rank = ds.len() as i64;
            let axis = if axis < 0 {
                (rank + axis) as usize
            } else {
                axis as usize
            };
            let outer: usize = ds[..axis].iter().product();
            let axis_size = ds[axis];
            let inner: usize = ds[axis + 1..].iter().product();
            let mut out_dims = [0usize; 8];
            let mut out_rank = 0;
            for &d in &ds[..axis] {
                out_dims[out_rank] = d;
                out_rank += 1;
            }
            for &d in is {
                out_dims[out_rank] = d;
                out_rank += 1;
            }
            for &d in &ds[axis + 1..] {
                out_dims[out_rank] = d;
                out_rank += 1;
            }
            Some(GatherPrecomp {
                axis,
                outer,
                axis_size,
                inner,
                out_dims,
                out_rank,
            })
        } else {
            None
        };

        Self {
            inputs,
            axis,
            precomp,
        }
    }
}

impl Layer for Gather {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let indices = get_tensor(values, &self.inputs[1])?;

        let rank = input.dims.len() as i64;
        let axis = if let Some(p) = &self.precomp {
            p.axis
        } else if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };
        let outer: usize = input.dims[..axis].iter().product();
        let axis_size = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();
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

        let num_indices = indices.numel();
        let numel = outer * num_indices * inner;

        let idx_is_int = indices.dtype() == DType::Int64;
        let resolve_idx = |i: usize| -> i64 {
            if idx_is_int {
                indices.ints()[i]
            } else {
                indices.floats()[i] as i64
            }
        };

        match input.dtype() {
            DType::Float => {
                let d = input.floats();
                let buf = output.as_mut_f32(numel);
                let mut dst = 0;
                for o in 0..outer {
                    for j in 0..num_indices {
                        let raw_idx = resolve_idx(j);
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
                    for j in 0..num_indices {
                        let raw_idx = resolve_idx(j);
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
            DType::String => unreachable!("strings not supported"),
        }
        let _ = axis; // suppress unused warning (used in precomp path)
        output.set_dims(&out_dims[..out_rank]);
        Ok(())
    }
}
