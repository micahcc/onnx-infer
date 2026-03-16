use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Hardmax {
    pub inputs: Vec<String>,
    pub axis: i64,
}

impl Hardmax {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for Hardmax {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let data = input.floats();
        let numel = data.len();
        let buf = output.as_mut_f32(numel);
        buf.fill(0.0);

        let axis_len = input.dims[axis];
        let outer: usize = input.dims[..axis].iter().product();
        let inner: usize = input.dims[axis + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for a in 0..axis_len {
                    let flat = o * axis_len * inner + a * inner + i;
                    let v = data[flat];
                    if v > best_val {
                        best_val = v;
                        best_idx = a;
                    }
                }
                let flat = o * axis_len * inner + best_idx * inner + i;
                buf[flat] = 1.0;
            }
        }

        output.set_dims(&input.dims);
        Ok(())
    }
}
