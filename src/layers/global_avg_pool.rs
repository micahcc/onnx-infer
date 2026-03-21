use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct GlobalAvgPoolPrecomp {
    pub n: usize,
    pub c: usize,
    pub spatial: usize,
    pub rank: usize,
}

pub struct GlobalAvgPool {
    pub inputs: Vec<String>,
    pub shape_cache: Dims,
    pub precomp: Option<GlobalAvgPoolPrecomp>,
}

impl GlobalAvgPool {
    pub fn compute_shapes(shape: &[usize]) -> GlobalAvgPoolPrecomp {
        GlobalAvgPoolPrecomp {
            n: shape[0],
            c: shape[1],
            spatial: shape[2..].iter().product(),
            rank: shape.len(),
        }
    }

    pub fn new(inputs: Vec<String>, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if initial_shape.len() >= 2 {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(initial_shape)),
            )
        } else {
            (Dims::new(), None)
        };
        Self {
            inputs,
            shape_cache,
            precomp,
        }
    }
}

impl Layer for GlobalAvgPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(&input.dims));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let input_f = input.floats().context("in GlobalAvgPool layer")?;
        let buf = output.as_mut_f32(p.n * p.c);
        for batch in 0..p.n {
            for ch in 0..p.c {
                let offset = (batch * p.c + ch) * p.spatial;
                let sum: f32 = input_f[offset..offset + p.spatial].iter().sum();
                buf[batch * p.c + ch] = sum / p.spatial as f32;
            }
        }

        let mut dims_buf = [1usize; 8];
        dims_buf[0] = p.n;
        dims_buf[1] = p.c;
        output.set_dims(&dims_buf[..p.rank]);
        Ok(())
    }
}
