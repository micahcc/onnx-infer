use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Tanh {
    pub inputs: Vec<String>,
}

impl Tanh {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Tanh {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats();
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = v.tanh();
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
