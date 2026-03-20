use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Log {
    pub inputs: Vec<String>,
}

impl Log {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Log {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in Log layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = v.ln();
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
