use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct QuantizeLinear {
    pub inputs: Vec<String>,
}

impl QuantizeLinear {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for QuantizeLinear {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let zero_point = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            get_tensor(values, &self.inputs[2])?
                .floats()
                .context("in QuantizeLinear layer")?[0]
        } else {
            0.0
        };
        let numel = input.numel();
        let buf = output.as_mut_f32(numel);
        crate::layers::quantize_u8_into(
            input.floats().context("in QuantizeLinear layer")?,
            scale.floats().context("in QuantizeLinear layer")?[0],
            zero_point,
            buf,
        );
        output.set_dims(&input.dims);
        Ok(())
    }
}
