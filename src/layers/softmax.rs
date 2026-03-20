use anyhow::Context;
use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct SoftmaxPrecomp {
    pub axis: usize,
    pub outer: usize,
    pub dim: usize,
    pub inner: usize,
}

pub struct Softmax {
    pub inputs: Vec<String>,
    pub axis: i64,
    pub shape_cache: Dims,
    pub precomp: Option<SoftmaxPrecomp>,
}

impl Softmax {
    pub fn compute_shapes(axis: i64, shape: &[usize]) -> SoftmaxPrecomp {
        let rank = shape.len() as i64;
        let axis = if axis < 0 {
            (rank + axis) as usize
        } else {
            axis as usize
        };
        SoftmaxPrecomp {
            axis,
            outer: shape[..axis].iter().product(),
            dim: shape[axis],
            inner: shape[axis + 1..].iter().product(),
        }
    }

    pub fn new(inputs: Vec<String>, axis: i64, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if !initial_shape.is_empty() {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(axis, initial_shape)),
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

impl Layer for Softmax {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(self.axis, &input.dims));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let inp = input.floats().context("in Softmax layer")?;
        let buf = output.as_mut_f32(inp.len());
        buf.copy_from_slice(inp);

        if p.inner == 1 {
            // Fast path: softmax on last axis (contiguous rows)
            for o in 0..p.outer {
                let row = &mut buf[o * p.dim..(o + 1) * p.dim];
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                let inv_sum = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv_sum;
                }
            }
        } else {
            for o in 0..p.outer {
                for i in 0..p.inner {
                    let mut max_val = f32::NEG_INFINITY;
                    for d in 0..p.dim {
                        let idx = (o * p.dim + d) * p.inner + i;
                        max_val = max_val.max(buf[idx]);
                    }
                    let mut sum = 0.0f32;
                    for d in 0..p.dim {
                        let idx = (o * p.dim + d) * p.inner + i;
                        buf[idx] = (buf[idx] - max_val).exp();
                        sum += buf[idx];
                    }
                    let inv_sum = 1.0 / sum;
                    for d in 0..p.dim {
                        let idx = (o * p.dim + d) * p.inner + i;
                        buf[idx] *= inv_sum;
                    }
                }
            }
        }

        output.set_dims(&input.dims);
        Ok(())
    }
}
