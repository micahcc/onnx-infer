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
    pub to: i64,
}

impl Cast {
    pub fn new(inputs: Vec<String>, to: i64) -> Self {
        Self { inputs, to }
    }
}

impl Layer for Cast {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let numel = input.numel();
        let to32 = self.to as i32;
        match to32 == ONNX_INT32 || to32 == ONNX_INT64 {
            true => {
                let buf = output.as_mut_i64(numel);
                match input.dtype() {
                    DType::Int64 => buf.copy_from_slice(input.ints()),
                    DType::Float => {
                        for (o, &v) in buf.iter_mut().zip(input.floats().iter()) {
                            *o = v as i64;
                        }
                    }
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
                }
                output.set_dims(&input.dims);
            }
        }
        Ok(())
    }
}
