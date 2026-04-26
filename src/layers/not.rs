use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Not {
    pub inputs: Vec<String>,
}

impl Not {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Not {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let data = input.ints().context("in Not layer")?;
        let buf = output.as_mut_i64(data.len());
        for (o, &v) in buf.iter_mut().zip(data.iter()) {
            *o = if v == 0 { 1 } else { 0 };
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
