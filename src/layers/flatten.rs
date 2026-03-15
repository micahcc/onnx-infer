use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Flatten {
    pub inputs: Vec<String>,
    pub axis: usize,
}

impl Flatten {
    pub fn new(inputs: Vec<String>, axis: usize) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for Flatten {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let outer: usize = input.dims[..self.axis].iter().product();
        let inner: usize = input.dims[self.axis..].iter().product();
        output.copy_from(input);
        output.dims = vec![outer, inner];
        Ok(())
    }
}
