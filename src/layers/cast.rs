use std::collections::HashMap;

use crate::DType;
use crate::ONNX_INT32;
use crate::ONNX_INT64;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Cast {
    pub inputs: Vec<String>,
    pub to_int: bool,
}

impl Cast {
    pub fn new(inputs: Vec<String>, to: i64) -> Self {
        let to32 = to as i32;
        Self {
            inputs,
            to_int: to32 == ONNX_INT32 || to32 == ONNX_INT64,
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
                    DType::Int64 => buf.copy_from_slice(input.ints()),
                    DType::Float => {
                        for (o, &v) in buf.iter_mut().zip(input.floats().iter()) {
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
                    DType::Float => buf.copy_from_slice(input.floats()),
                    DType::Int64 => {
                        for (o, &v) in buf.iter_mut().zip(input.ints().iter()) {
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
