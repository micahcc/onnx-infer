use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Softmax {
    pub inputs: Vec<String>,
    pub axis: i64,
    // Precomputed (0 = not precomputed)
    pub pre_axis: usize,
    pub pre_outer: usize,
    pub pre_dim: usize,
    pub pre_inner: usize,
}

impl Softmax {
    pub fn new(inputs: Vec<String>, axis: i64, input_shape: &[usize]) -> Self {
        let mut s = Self {
            inputs,
            axis,
            pre_axis: 0,
            pre_outer: 0,
            pre_dim: 0,
            pre_inner: 0,
        };
        if !input_shape.is_empty() {
            let shape = input_shape;
            s.precompute(shape);
        }
        s
    }

    fn precompute(&mut self, shape: &[usize]) {
        let rank = shape.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };
        self.pre_axis = axis;
        self.pre_outer = shape[..axis].iter().product();
        self.pre_dim = shape[axis];
        self.pre_inner = shape[axis + 1..].iter().product();
    }
}

impl Layer for Softmax {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let (outer, dim, inner) = if self.pre_dim > 0 {
            (self.pre_outer, self.pre_dim, self.pre_inner)
        } else {
            let rank = input.dims.len() as i64;
            let axis = if self.axis < 0 {
                (rank + self.axis) as usize
            } else {
                self.axis as usize
            };
            let outer: usize = input.dims[..axis].iter().product();
            let dim = input.dims[axis];
            let inner: usize = input.dims[axis + 1..].iter().product();
            (outer, dim, inner)
        };

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

        output.set_dims(&input.dims);
        Ok(())
    }
}
