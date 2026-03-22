use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct And {
    pub inputs: Vec<String>,
}

impl Layer for And {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;
        let a_data = a.ints().context("in And layer")?;
        let b_data = b.ints().context("in And layer")?;
        let numel = a_data.len();
        let buf = output.as_mut_i64(numel);
        for (i, (av, bv)) in a_data.iter().zip(b_data.iter()).enumerate() {
            buf[i] = if *av != 0 && *bv != 0 { 1 } else { 0 };
        }
        output.set_dims(&a.dims);
        Ok(())
    }
}
