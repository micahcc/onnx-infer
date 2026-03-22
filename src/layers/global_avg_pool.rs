use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Layout;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct GlobalAvgPoolPrecomp {
    pub n: usize,
    pub c: usize,
    pub spatial: usize,
    pub rank: usize,
}

#[derive(Debug)]
pub struct GlobalAvgPool {
    pub inputs: Vec<String>,
    pub nhwc: bool,
    pub shape_cache: Dims,
    pub precomp: Option<GlobalAvgPoolPrecomp>,
}

impl GlobalAvgPool {
    pub fn compute_shapes(shape: &[usize], nhwc: bool) -> GlobalAvgPoolPrecomp {
        assert!(nhwc, "GlobalAvgPool::compute_shapes requires NHWC");
        GlobalAvgPoolPrecomp {
            n: shape[0],
            c: shape[shape.len() - 1],
            spatial: shape[1..shape.len() - 1].iter().product(),
            rank: shape.len(),
        }
    }

    pub fn new(inputs: Vec<String>, initial_shape: &[usize], nhwc: bool) -> Self {
        let (shape_cache, precomp) = if initial_shape.len() >= 2 {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(initial_shape, nhwc)),
            )
        } else {
            (Dims::new(), None)
        };
        Self {
            inputs,
            nhwc,
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
                self.precomp = Some(Self::compute_shapes(&input.dims, self.nhwc));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let input_f = input.floats().context("in GlobalAvgPool layer")?;
        let buf = output.as_mut_f32(p.n * p.c);

        assert!(self.nhwc, "GlobalAvgPool::execute requires NHWC input layout");

        // NHWC: input[batch][h][w][c] — channels interleaved at each spatial position
        buf.fill(0.0);
        for batch in 0..p.n {
            let in_batch = batch * p.spatial * p.c;
            let out_batch = batch * p.c;
            for s in 0..p.spatial {
                let in_off = in_batch + s * p.c;
                for ch in 0..p.c {
                    buf[out_batch + ch] += input_f[in_off + ch];
                }
            }
            let inv = 1.0 / p.spatial as f32;
            for ch in 0..p.c {
                buf[out_batch + ch] *= inv;
            }
        }
        // Output is [N, 1, 1, C] for NHWC
        let mut dims_buf = [1usize; 8];
        dims_buf[0] = p.n;
        dims_buf[p.rank - 1] = p.c;
        output.set_dims(&dims_buf[..p.rank]);
        output.layout = Layout::NHWC;

        Ok(())
    }
}
