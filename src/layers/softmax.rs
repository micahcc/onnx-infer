use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Softmax {
    pub inputs: Vec<String>,
    pub axis: i64,
}

impl Softmax {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for Softmax {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        let outer: usize = input.dims[..axis].iter().product();
        let dim = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();

        let inp = input.floats();
        let buf = output.as_mut_f32(inp.len());
        buf.copy_from_slice(inp);

        for o in 0..outer {
            for i in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim {
                    let idx = (o * dim + d) * inner + i;
                    max_val = max_val.max(buf[idx]);
                }
                let mut sum = 0.0f32;
                for d in 0..dim {
                    let idx = (o * dim + d) * inner + i;
                    buf[idx] = (buf[idx] - max_val).exp();
                    sum += buf[idx];
                }
                for d in 0..dim {
                    let idx = (o * dim + d) * inner + i;
                    buf[idx] /= sum;
                }
            }
        }

        output.dims.clone_from(&input.dims);
        Ok(())
    }
}
