use anyhow::Context;
use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct DequantizePrecomp {
    pub axis: usize,
    pub outer: usize,
    pub ch: usize,
    pub inner: usize,
}

pub struct DequantizeLinear {
    pub inputs: Vec<String>,
    pub axis: i64,
    default_zp: Tensor,
    pub shape_cache: Dims,
    pub precomp: Option<DequantizePrecomp>,
}

impl DequantizeLinear {
    pub fn compute_shapes(axis: i64, shape: &[usize]) -> Option<DequantizePrecomp> {
        let rank = shape.len() as i64;
        let a = if axis < 0 {
            (rank + axis) as usize
        } else {
            axis as usize
        };
        if a >= shape.len() {
            return None;
        }
        Some(DequantizePrecomp {
            axis: a,
            outer: shape[..a].iter().product(),
            ch: shape[a],
            inner: shape[a + 1..].iter().product(),
        })
    }

    pub fn new(inputs: Vec<String>, axis: i64, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if !initial_shape.is_empty() {
            (
                Dims::from_slice(initial_shape),
                Self::compute_shapes(axis, initial_shape),
            )
        } else {
            (Dims::new(), None)
        };

        Self {
            inputs,
            axis,
            default_zp: Tensor::new(crate::dims![], vec![0.0]),
            shape_cache,
            precomp,
        }
    }
}

impl Layer for DequantizeLinear {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let zp = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            get_tensor(values, &self.inputs[2])?
        } else {
            &self.default_zp
        };

        let numel = input.numel();
        let scale_f = scale.floats().context("in DequantizeLinear layer")?;
        let input_f = input.floats().context("in DequantizeLinear layer")?;
        let zp_f = zp.floats().context("in DequantizeLinear layer")?;

        if scale_f.len() == 1 {
            let s = scale_f[0];
            let z = zp_f[0];
            let buf = output.as_mut_f32(numel);
            for i in 0..numel {
                buf[i] = (input_f[i] - z) * s;
            }
            output.set_dims(&input.dims);
        } else {
            let p = match &self.precomp {
                Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
                _ => {
                    self.precomp = Self::compute_shapes(self.axis, &input.dims);
                    self.shape_cache.clone_from(&input.dims);
                    self.precomp.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "DequantizeLinear: axis {} out of bounds for shape {:?}",
                            self.axis,
                            input.dims
                        )
                    })?
                }
            };

            let buf = output.as_mut_f32(numel);
            for o in 0..p.outer {
                for c in 0..p.ch {
                    let s = scale_f[c];
                    let z = zp_f[c];
                    let base = (o * p.ch + c) * p.inner;
                    for i in 0..p.inner {
                        buf[base + i] = (input_f[base + i] - z) * s;
                    }
                }
            }
            output.set_dims(&input.dims);
        }
        Ok(())
    }
}
