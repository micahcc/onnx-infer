use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct GlobalAvgPool {
    pub inputs: Vec<String>,
    // Precomputed (0 = not precomputed)
    pub pre_n: usize,
    pub pre_c: usize,
    pub pre_spatial: usize,
    pub pre_rank: usize,
}

impl GlobalAvgPool {
    pub fn new(inputs: Vec<String>, input_shape: &[usize]) -> Self {
        let mut s = Self {
            inputs,
            pre_n: 0,
            pre_c: 0,
            pre_spatial: 0,
            pre_rank: 0,
        };
        if !input_shape.is_empty() {
            let shape = input_shape;
            if shape.len() >= 2 {
                s.pre_n = shape[0];
                s.pre_c = shape[1];
                s.pre_spatial = shape[2..].iter().product();
                s.pre_rank = shape.len();
            }
        }
        s
    }
}

impl Layer for GlobalAvgPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let (n, c, spatial, rank) = if self.pre_rank > 0 {
            (self.pre_n, self.pre_c, self.pre_spatial, self.pre_rank)
        } else {
            (
                input.dims[0],
                input.dims[1],
                input.dims[2..].iter().product(),
                input.dims.len(),
            )
        };

        let input_f = input.floats();
        let buf = output.as_mut_f32(n * c);
        for batch in 0..n {
            for ch in 0..c {
                let offset = (batch * c + ch) * spatial;
                let sum: f32 = input_f[offset..offset + spatial].iter().sum();
                buf[batch * c + ch] = sum / spatial as f32;
            }
        }

        let mut dims_buf = [1usize; 8];
        dims_buf[0] = n;
        dims_buf[1] = c;
        output.set_dims(&dims_buf[..rank]);
        Ok(())
    }
}
