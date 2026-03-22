use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Round {
    pub inputs: Vec<String>,
}

impl Round {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Round {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in Round layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            let rounded = v.round();
            // ONNX Round uses "round half to even" (banker's rounding)
            *o = if (v - v.floor() - 0.5).abs() < f32::EPSILON {
                if rounded as i64 % 2 != 0 {
                    rounded - 1.0
                } else {
                    rounded
                }
            } else {
                rounded
            };
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
