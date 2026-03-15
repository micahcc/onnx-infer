use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct BatchNorm {
    pub inputs: Vec<String>,
    pub epsilon: f32,
}

impl BatchNorm {
    pub fn new(inputs: Vec<String>, epsilon: f32) -> Self {
        Self { inputs, epsilon }
    }
}

impl Layer for BatchNorm {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let bias = get_tensor(values, &self.inputs[2])?;
        let mean = get_tensor(values, &self.inputs[3])?;
        let var = get_tensor(values, &self.inputs[4])?;

        let n = input.dims[0];
        let c = input.dims[1];
        let spatial: usize = input.dims[2..].iter().product();

        let input_f = input.floats();
        let scale_f = scale.floats();
        let bias_f = bias.floats();
        let mean_f = mean.floats();
        let var_f = var.floats();

        let total = input_f.len();
        let buf = output.as_mut_f32(total);
        buf.copy_from_slice(input_f);

        for batch in 0..n {
            for ch in 0..c {
                let s = scale_f[ch];
                let b = bias_f[ch];
                let m = mean_f[ch];
                let v = var_f[ch];
                let inv_std = 1.0 / (v + self.epsilon).sqrt();
                let base = (batch * c + ch) * spatial;
                for i in 0..spatial {
                    buf[base + i] = (buf[base + i] - m) * inv_std * s + b;
                }
            }
        }

        output.dims.clone_from(&input.dims);
        Ok(())
    }
}
