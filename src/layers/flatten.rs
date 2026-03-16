use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Flatten {
    pub inputs: Vec<String>,
    pub axis: usize,
    // Precomputed (0 = not precomputed)
    pub pre_outer: usize,
    pub pre_inner: usize,
}

impl Flatten {
    pub fn new(inputs: Vec<String>, axis: usize, input_shape: &[usize]) -> Self {
        let mut s = Self {
            inputs,
            axis,
            pre_outer: 0,
            pre_inner: 0,
        };
        if !input_shape.is_empty() {
            let shape = input_shape;
            s.pre_outer = shape[..axis].iter().product();
            s.pre_inner = shape[axis..].iter().product();
        }
        s
    }
}

impl Layer for Flatten {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let (outer, inner) = if self.pre_inner > 0 {
            (self.pre_outer, self.pre_inner)
        } else {
            (
                input.dims[..self.axis].iter().product(),
                input.dims[self.axis..].iter().product(),
            )
        };
        output.copy_from(input);
        output.set_dims(&[outer, inner]);
        Ok(())
    }
}
