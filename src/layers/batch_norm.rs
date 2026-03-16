use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct BatchNormPrecomp {
    pub n: usize,
    pub c: usize,
    pub spatial: usize,
}

pub struct BatchNorm {
    pub inputs: Vec<String>,
    pub epsilon: f32,
    pub shape_cache: Dims,
    pub precomp: Option<BatchNormPrecomp>,
}

impl BatchNorm {
    pub fn compute_shapes(shape: &[usize]) -> BatchNormPrecomp {
        BatchNormPrecomp {
            n: shape[0],
            c: shape[1],
            spatial: shape[2..].iter().product(),
        }
    }

    pub fn new(inputs: Vec<String>, epsilon: f32, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if initial_shape.len() >= 2 {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(initial_shape)),
            )
        } else {
            (Dims::new(), None)
        };
        Self {
            inputs,
            epsilon,
            shape_cache,
            precomp,
        }
    }
}

impl Layer for BatchNorm {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let bias = get_tensor(values, &self.inputs[2])?;
        let mean = get_tensor(values, &self.inputs[3])?;
        let var = get_tensor(values, &self.inputs[4])?;

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(&input.dims));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let input_f = input.floats();
        let scale_f = scale.floats();
        let bias_f = bias.floats();
        let mean_f = mean.floats();
        let var_f = var.floats();

        let total = input_f.len();
        let buf = output.as_mut_f32(total);
        buf.copy_from_slice(input_f);

        for batch in 0..p.n {
            for ch in 0..p.c {
                let s = scale_f[ch];
                let b = bias_f[ch];
                let m = mean_f[ch];
                let v = var_f[ch];
                let inv_std = 1.0 / (v + self.epsilon).sqrt();
                let base = (batch * p.c + ch) * p.spatial;
                for i in 0..p.spatial {
                    buf[base + i] = (buf[base + i] - m) * inv_std * s + b;
                }
            }
        }

        output.set_dims(&input.dims);
        Ok(())
    }
}
