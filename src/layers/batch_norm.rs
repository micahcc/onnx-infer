use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct BatchNorm {
    pub inputs: Vec<String>,
    pub epsilon: f32,
    // Precomputed (0 = not precomputed)
    pub pre_n: usize,
    pub pre_c: usize,
    pub pre_spatial: usize,
}

impl BatchNorm {
    pub fn new(inputs: Vec<String>, epsilon: f32, input_shape: &[usize]) -> Self {
        let mut s = Self {
            inputs,
            epsilon,
            pre_n: 0,
            pre_c: 0,
            pre_spatial: 0,
        };
        if !input_shape.is_empty() {
            let shape = input_shape;
            if shape.len() >= 2 {
                s.pre_n = shape[0];
                s.pre_c = shape[1];
                s.pre_spatial = shape[2..].iter().product();
            }
        }
        s
    }
}

impl Layer for BatchNorm {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let bias = get_tensor(values, &self.inputs[2])?;
        let mean = get_tensor(values, &self.inputs[3])?;
        let var = get_tensor(values, &self.inputs[4])?;

        let (n, c, spatial) = if self.pre_spatial > 0 {
            (self.pre_n, self.pre_c, self.pre_spatial)
        } else {
            (
                input.dims[0],
                input.dims[1],
                input.dims[2..].iter().product(),
            )
        };

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

        output.set_dims(&input.dims);
        Ok(())
    }
}
