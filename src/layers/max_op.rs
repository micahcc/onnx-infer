use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Max {
    pub inputs: Vec<String>,
}

impl Max {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Max {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let first = get_tensor(values, &self.inputs[0])?;
        let numel = first.numel();
        let dims = first.dims.clone();

        match first.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                buf.copy_from_slice(first.floats());
                for name in &self.inputs[1..] {
                    let other = get_tensor(values, name)?;
                    let of = other.floats();
                    for (o, &v) in buf.iter_mut().zip(of.iter()) {
                        *o = o.max(v);
                    }
                }
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(numel);
                buf.copy_from_slice(first.ints());
                for name in &self.inputs[1..] {
                    let other = get_tensor(values, name)?;
                    let of = other.ints();
                    for (o, &v) in buf.iter_mut().zip(of.iter()) {
                        *o = (*o).max(v);
                    }
                }
            }
        }

        output.set_dims(&dims);
        Ok(())
    }
}
