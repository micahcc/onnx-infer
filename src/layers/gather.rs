use anyhow::Context;
use std::collections::HashMap;

use crate::DType;
use crate::Dims;
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
    pub shape_cache: Dims,
    pub precomp: Option<GatherPrecomp>,
}

impl Gather {
    pub fn compute_shapes(
        axis_raw: i64,
        data_shape: &[usize],
        indices_shape: &[usize],
    ) -> GatherPrecomp {
        let rank = data_shape.len() as i64;
        let axis = if axis_raw < 0 {
            (rank + axis_raw) as usize
        } else {
            axis_raw as usize
        };
        let outer: usize = data_shape[..axis].iter().product();
        let axis_size = data_shape[axis];
        let inner: usize = data_shape[axis + 1..].iter().product();
        let mut out_dims = [0usize; 8];
        let mut out_rank = 0;
        for &d in &data_shape[..axis] {
            out_dims[out_rank] = d;
            out_rank += 1;
        }
        for &d in indices_shape {
            out_dims[out_rank] = d;
            out_rank += 1;
        }
        for &d in &data_shape[axis + 1..] {
            out_dims[out_rank] = d;
            out_rank += 1;
        }
        GatherPrecomp {
            axis,
            outer,
            axis_size,
            inner,
            out_dims,
            out_rank,
        }
    }

    pub fn new(
        inputs: Vec<String>,
        axis: i64,
        initial_data_shape: &[usize],
        initial_indices_shape: &[usize],
    ) -> Self {
        let (shape_cache, precomp) =
            if !initial_data_shape.is_empty() && !initial_indices_shape.is_empty() {
                let mut cache = Dims::from_slice(initial_data_shape);
                cache.extend_from_slice(initial_indices_shape);
                (
                    cache,
                    Some(Self::compute_shapes(
                        axis,
                        initial_data_shape,
                        initial_indices_shape,
                    )),
                )
            } else {
                (Dims::new(), None)
            };

        Self {
            inputs,
            axis,
            shape_cache,
            precomp,
        }
    }
}

impl Layer for Gather {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let indices = get_tensor(values, &self.inputs[1])?;

        let mut key = Dims::from_slice(&input.dims);
        key.extend_from_slice(&indices.dims);
        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == key.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(self.axis, &input.dims, &indices.dims));
                self.shape_cache = key;
                self.precomp.as_ref().expect("just set")
            }
        };

        let num_indices = indices.numel();
        let numel = p.outer * num_indices * p.inner;

        let idx_is_int = indices.dtype() == DType::Int64;
        let resolve_idx = |i: usize| -> anyhow::Result<i64> {
            use anyhow::Context;
            if idx_is_int {
                Ok(indices.ints().context("Gather: indices")?[i])
            } else {
                Ok(indices.floats().context("Gather: indices")?[i] as i64)
            }
        };

        match input.dtype() {
            DType::Float => {
                let d = input.floats().context("in Gather layer")?;
                let buf = output.as_mut_f32(numel);
                let mut dst = 0;
                for o in 0..p.outer {
                    for j in 0..num_indices {
                        let raw_idx = resolve_idx(j).context("in Gather layer: resolving index")?;
                        let idx = if raw_idx < 0 {
                            (p.axis_size as i64 + raw_idx) as usize
                        } else {
                            raw_idx as usize
                        };
                        let base = o * p.axis_size * p.inner + idx * p.inner;
                        buf[dst..dst + p.inner].copy_from_slice(&d[base..base + p.inner]);
                        dst += p.inner;
                    }
                }
            }
            DType::Int64 => {
                let d = input.ints().context("in Gather layer")?;
                let buf = output.as_mut_i64(numel);
                let mut dst = 0;
                for o in 0..p.outer {
                    for j in 0..num_indices {
                        let raw_idx = resolve_idx(j).context("in Gather layer: resolving index")?;
                        let idx = if raw_idx < 0 {
                            (p.axis_size as i64 + raw_idx) as usize
                        } else {
                            raw_idx as usize
                        };
                        let base = o * p.axis_size * p.inner + idx * p.inner;
                        buf[dst..dst + p.inner].copy_from_slice(&d[base..base + p.inner]);
                        dst += p.inner;
                    }
                }
            }
            DType::String => unreachable!("strings not supported"),
        }
        output.set_dims(&p.out_dims[..p.out_rank]);
        Ok(())
    }
}
