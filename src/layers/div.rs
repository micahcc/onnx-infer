use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Div {
    pub inputs: Vec<String>,
    pub legacy_broadcast: bool,
    pub axis: usize,
}

impl Div {
    pub fn new(inputs: Vec<String>, legacy_broadcast: bool, axis: usize) -> Self {
        Self {
            inputs,
            legacy_broadcast,
            axis,
        }
    }
}

impl Layer for Div {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;
        crate::layers::binary_op(a, b, output, self.legacy_broadcast, self.axis, |a, b| a / b)
    }
}
