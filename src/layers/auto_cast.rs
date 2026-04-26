
use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::Values;

#[derive(Debug)]
pub struct AutoCastF32 {
    pub inputs: Vec<String>,
}

impl AutoCastF32 {
    pub fn new(input: String) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl Layer for AutoCastF32 {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let src = get_tensor(values, &self.inputs[0])?;
        output.copy_cast_f32(src).context("in AutoCast layer")?;
        Ok(())
    }
}
