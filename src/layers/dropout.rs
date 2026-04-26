use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

/// Dropout is a no-op during inference.
#[derive(Debug)]
pub struct Dropout {
    pub inputs: Vec<String>,
}

impl Dropout {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Dropout {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        output.copy_from(input);
        Ok(())
    }
}
