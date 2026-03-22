use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct SoftmaxPrecomp {
    pub axis: usize,
    pub outer: usize,
    pub dim: usize,
    pub inner: usize,
}

#[derive(Debug)]
pub struct Softmax {
    pub inputs: Vec<String>,
    pub axis: i64,
    /// For opset < 13, Softmax coerces to 2D: dims [0..axis) → outer, [axis..) → inner
    pub coerce_2d: bool,
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

    pub fn new(inputs: Vec<String>, axis: i64, coerce_2d: bool, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if !initial_shape.is_empty() {
            let effective_shape = if coerce_2d {
                Self::coerced_shape(axis, initial_shape)
            } else {
                initial_shape.to_vec()
            };
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(if coerce_2d { 1 } else { axis }, &effective_shape)),
            )
        } else {
            (Dims::new(), None)
        };
        Self {
            inputs,
            axis,
            coerce_2d,
            shape_cache,
            precomp,
        }
    }

    /// For opset < 13: coerce to 2D shape [product(0..axis), product(axis..)]
    fn coerced_shape(axis: i64, shape: &[usize]) -> Vec<usize> {
        let rank = shape.len() as i64;
        let axis = if axis < 0 { (rank + axis) as usize } else { axis as usize };
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis..].iter().product();
        vec![outer, inner]
    }
}

impl Layer for Softmax {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                let effective = if self.coerce_2d {
                    let c = Self::coerced_shape(self.axis, &input.dims);
                    Self::compute_shapes(1, &c)
                } else {
                    Self::compute_shapes(self.axis, &input.dims)
                };
                self.precomp = Some(effective);
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
