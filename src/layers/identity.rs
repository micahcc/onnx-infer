use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Identity {
    pub inputs: Vec<String>,
}

impl Identity {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Identity {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        output.copy_from(input);
        Ok(())
    }
}
