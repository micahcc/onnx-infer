use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct ConstantOfShape {
    pub inputs: Vec<String>,
    pub fill_f32: f32,
    pub fill_i64: i64,
    pub dtype: DType,
}

impl ConstantOfShape {
    pub fn new(inputs: Vec<String>, fill_f32: f32, fill_i64: i64, dtype: DType) -> Self {
        Self {
            inputs,
            fill_f32,
            fill_i64,
            dtype,
        }
    }
}

impl Layer for ConstantOfShape {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let shape_tensor = get_tensor(values, &self.inputs[0])?;
        let mut shape = [0usize; 8];
        let shape_len = shape_tensor.numel();
        match shape_tensor.dtype() {
            DType::Int64 => {
                for (i, &v) in shape_tensor.ints().iter().enumerate() {
                    shape[i] = v as usize;
                }
            }
            DType::Float => {
                for (i, &v) in shape_tensor.floats().iter().enumerate() {
                    shape[i] = v as usize;
                }
            }
            DType::String => unreachable!("strings not supported"),
        }
        let numel: usize = shape[..shape_len].iter().product();

        match self.dtype {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                buf.fill(self.fill_f32);
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(numel);
                buf.fill(self.fill_i64);
            }
            DType::String => unreachable!("strings not supported"),
        }
        output.set_dims(&shape[..shape_len]);
        Ok(())
    }
}
