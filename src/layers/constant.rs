use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Constant {
    pub value: Tensor,
}

impl Constant {
    pub fn new(value: Tensor) -> Self {
        Self { value }
    }
}

impl Layer for Constant {
    fn execute(&mut self, _values: &Values, output: &mut Tensor) -> Result<()> {
        output.copy_from(&self.value);
        Ok(())
    }
}
