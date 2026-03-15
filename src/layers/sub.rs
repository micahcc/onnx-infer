use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Sub {
    pub inputs: Vec<String>,
    pub legacy_broadcast: bool,
    pub axis: usize,
    a: Tensor,
    b: Tensor,
}

impl Sub {
    pub fn new(inputs: Vec<String>, legacy_broadcast: bool, axis: usize) -> Self {
        Self {
            inputs,
            legacy_broadcast,
            axis,
            a: Tensor::default(),
            b: Tensor::default(),
        }
    }
}

impl Layer for Sub {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let src_a = get_tensor(values, &self.inputs[0])?;
        let src_b = get_tensor(values, &self.inputs[1])?;
        self.a.copy_cast_f32(src_a);
        self.b.copy_cast_f32(src_b);
        crate::layers::binary_op(
            &self.a,
            &self.b,
            output,
            self.legacy_broadcast,
            self.axis,
            |a, b| a - b,
        )
    }
}
