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
        let shape: Vec<usize> = match shape_tensor.dtype() {
            DType::Int64 => shape_tensor.ints().iter().map(|&v| v as usize).collect(),
            DType::Float => shape_tensor.floats().iter().map(|&v| v as usize).collect(),
        };
        let numel: usize = shape.iter().product();

        match self.dtype {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                buf.fill(self.fill_f32);
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(numel);
                buf.fill(self.fill_i64);
            }
        }
        output.set_dims(&shape);
        Ok(())
    }
}
