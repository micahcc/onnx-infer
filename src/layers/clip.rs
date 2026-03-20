use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Clip {
    pub inputs: Vec<String>,
    pub attr_min: f32,
    pub attr_max: f32,
}

impl Clip {
    pub fn new(inputs: Vec<String>, attr_min: f32, attr_max: f32) -> Self {
        Self {
            inputs,
            attr_min,
            attr_max,
        }
    }
}

impl Layer for Clip {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let min_val = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            get_tensor(values, &self.inputs[1])?.floats().context("in Clip layer")?[0]
        } else {
            self.attr_min
        };
        let max_val = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            get_tensor(values, &self.inputs[2])?.floats().context("in Clip layer")?[0]
        } else {
            self.attr_max
        };

        let inp = input.floats().context("in Clip layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = v.clamp(min_val, max_val);
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
