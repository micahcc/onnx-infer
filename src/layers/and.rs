use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct And {
    pub inputs: Vec<String>,
}

fn as_bool_slice(t: &Tensor) -> Vec<bool> {
    if let Ok(v) = t.ints() {
        v.iter().map(|&x| x != 0).collect()
    } else if let Ok(v) = t.floats() {
        v.iter().map(|&x| x != 0.0).collect()
    } else {
        vec![]
    }
}

impl Layer for And {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;
        let a_data = as_bool_slice(a);
        let b_data = as_bool_slice(b);
        let numel = a_data.len().min(b_data.len());
        let buf = output.as_mut_i64(numel);
        for i in 0..numel {
            buf[i] = if a_data[i] && b_data[i] { 1 } else { 0 };
        }
        output.set_dims(&a.dims);
        Ok(())
    }
}
