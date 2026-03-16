use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

/// Sum: element-wise sum of all input tensors (must be same shape).
pub struct Sum {
    pub inputs: Vec<String>,
}

impl Sum {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Sum {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let first = get_tensor(values, &self.inputs[0])?;
        let numel = first.numel();
        let out_dims = first.dims.clone();
        let buf = output.as_mut_f32(numel);
        buf.copy_from_slice(first.floats());

        for name in &self.inputs[1..] {
            let t = get_tensor(values, name)?;
            let f = t.floats();
            for (o, &v) in buf.iter_mut().zip(f.iter()) {
                *o += v;
            }
        }
        output.set_dims(&out_dims);
        Ok(())
    }
}
