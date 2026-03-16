use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Range {
    pub inputs: Vec<String>,
}

impl Range {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Range {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let start = get_tensor(values, &self.inputs[0])?;
        let limit = get_tensor(values, &self.inputs[1])?;
        let delta = get_tensor(values, &self.inputs[2])?;

        match start.dtype() {
            DType::Float => {
                let s = start.floats()[0];
                let l = limit.floats()[0];
                let d = delta.floats()[0];
                let n = ((l - s) / d).ceil().max(0.0) as usize;
                let buf = output.as_mut_f32(n);
                for (i, v) in buf.iter_mut().enumerate() {
                    *v = s + (i as f32) * d;
                }
                output.set_dims(&[n]);
            }
            DType::Int64 => {
                let s = start.ints()[0];
                let l = limit.ints()[0];
                let d = delta.ints()[0];
                let n = if d > 0 {
                    ((l - s + d - 1) / d).max(0) as usize
                } else if d < 0 {
                    ((s - l - d - 1) / (-d)).max(0) as usize
                } else {
                    0
                };
                let buf = output.as_mut_i64(n);
                for (i, v) in buf.iter_mut().enumerate() {
                    *v = s + (i as i64) * d;
                }
                output.set_dims(&[n]);
            }
        }

        Ok(())
    }
}
