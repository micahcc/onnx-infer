use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct GlobalAvgPool {
    pub inputs: Vec<String>,
}

impl GlobalAvgPool {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for GlobalAvgPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let n = input.dims[0];
        let c = input.dims[1];
        let spatial: usize = input.dims[2..].iter().product();

        let input_f = input.floats();
        let buf = output.as_mut_f32(n * c);
        for batch in 0..n {
            for ch in 0..c {
                let offset = (batch * c + ch) * spatial;
                let sum: f32 = input_f[offset..offset + spatial].iter().sum();
                buf[batch * c + ch] = sum / spatial as f32;
            }
        }

        let rank = input.dims.len();
        let mut dims_buf = [1usize; 8];
        dims_buf[0] = n;
        dims_buf[1] = c;
        output.set_dims(&dims_buf[..rank]);
        Ok(())
    }
}
