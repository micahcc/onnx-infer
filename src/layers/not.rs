use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Not {
    pub inputs: Vec<String>,
}

impl Not {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Not {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        // Accept both int and float boolean representations (IsNaN/IsInf output floats).
        let numel = input.dims.iter().product::<usize>();
        let buf = output.as_mut_i64(numel);
        if let Ok(data) = input.ints() {
            for (o, &v) in buf.iter_mut().zip(data.iter()) {
                *o = if v == 0 { 1 } else { 0 };
            }
        } else if let Ok(data) = input.floats() {
            for (o, &v) in buf.iter_mut().zip(data.iter()) {
                *o = if v == 0.0 { 1 } else { 0 };
            }
        } else {
            anyhow::bail!("Not: unsupported input type");
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
