use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Min {
    pub inputs: Vec<String>,
}

impl Min {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Min {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let first = get_tensor(values, &self.inputs[0])?;
        let numel = first.numel();
        let rank = first.dims.len();
        let mut dims = [0usize; 8];
        dims[..rank].copy_from_slice(&first.dims);

        match first.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                buf.copy_from_slice(first.floats().context("in Min layer")?);
                for name in &self.inputs[1..] {
                    let other = get_tensor(values, name)?;
                    let of = other.floats().context("in Min layer")?;
                    for (o, &v) in buf.iter_mut().zip(of.iter()) {
                        *o = o.min(v);
                    }
                }
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(numel);
                buf.copy_from_slice(first.ints().context("in Min layer")?);
                for name in &self.inputs[1..] {
                    let other = get_tensor(values, name)?;
                    let of = other.ints().context("in Min layer")?;
                    for (o, &v) in buf.iter_mut().zip(of.iter()) {
                        *o = (*o).min(v);
                    }
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        output.set_dims(&dims[..rank]);
        Ok(())
    }
}
