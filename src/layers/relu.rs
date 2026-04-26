use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Relu {
    pub inputs: Vec<String>,
}

impl Relu {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Relu {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in Relu layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = v.max(0.0);
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
