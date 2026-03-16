use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct LeakyRelu {
    pub inputs: Vec<String>,
    pub alpha: f32,
}

impl LeakyRelu {
    pub fn new(inputs: Vec<String>, alpha: f32) -> Self {
        Self { inputs, alpha }
    }
}

impl Layer for LeakyRelu {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats();
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = if v >= 0.0 { v } else { self.alpha * v };
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
