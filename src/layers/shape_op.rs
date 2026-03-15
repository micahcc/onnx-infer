use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Shape {
    pub inputs: Vec<String>,
}

impl Shape {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Shape {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();
        let buf = output.as_mut_i64(rank);
        for (o, &d) in buf.iter_mut().zip(input.dims.iter()) {
            *o = d as i64;
        }
        output.set_dims(&[rank]);
        Ok(())
    }
}
