use anyhow::Context;
use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::onnx_ir::ElemType;

pub struct Cast {
    pub inputs: Vec<String>,
    pub to_int: bool,
}

impl Cast {
    pub fn new(inputs: Vec<String>, to: i64) -> Self {
        Self {
            inputs,
            to_int: ElemType::from_onnx(to as i32).is_int(),
        }
    }
}

impl Layer for Cast {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let numel = input.numel();
        match self.to_int {
            true => {
                let buf = output.as_mut_i64(numel);
                match input.dtype() {
                    DType::Int64 => buf.copy_from_slice(input.ints().context("in Cast layer")?),
                    DType::Float => {
                        for (o, &v) in buf.iter_mut().zip(input.floats().context("in Cast layer")?.iter()) {
                            *o = v as i64;
                        }
                    }
                    DType::String => unreachable!("strings not supported"),
                }
                output.set_dims(&input.dims);
            }
            false => {
                let buf = output.as_mut_f32(numel);
                match input.dtype() {
                    DType::Float => buf.copy_from_slice(input.floats().context("in Cast layer")?),
                    DType::Int64 => {
                        for (o, &v) in buf.iter_mut().zip(input.ints().context("in Cast layer")?.iter()) {
                            *o = v as f32;
                        }
                    }
                    DType::String => unreachable!("strings not supported"),
                }
                output.set_dims(&input.dims);
            }
        }
        Ok(())
    }
}
