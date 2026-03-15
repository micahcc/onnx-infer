use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::layers::Layer;

pub struct Constant {
    pub value: Tensor,
}

impl Constant {
    pub fn new(value: Tensor) -> Self {
        Self { value }
    }
}

impl Layer for Constant {
    fn execute(&mut self, _values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        output.copy_from(&self.value);
        Ok(())
    }
}
