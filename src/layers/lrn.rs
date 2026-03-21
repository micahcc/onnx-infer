use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Lrn {
    pub inputs: Vec<String>,
    pub size: usize,
    pub alpha: f32,
    pub beta: f32,
    pub bias: f32,
}

impl Lrn {
    pub fn new(inputs: Vec<String>, size: usize, alpha: f32, beta: f32, bias: f32) -> Self {
        Self {
            inputs,
            size,
            alpha,
            beta,
            bias,
        }
    }
}

impl Layer for Lrn {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in Lrn layer")?;
        let dims = &input.dims;

        anyhow::ensure!(
            dims.len() >= 2,
            "Lrn: input must have at least 2 dimensions"
        );

        let n = dims[0];
        let c = dims[1];
        let spatial: usize = dims[2..].iter().product();
        let numel = inp.len();

        let buf = output.as_mut_f32(numel);

        let half_size = (self.size - 1) / 2;
        // ceil((size-1)/2) — for even size this differs from floor
        let half_size_ceil = (self.size - 1).div_ceil(2);

        for batch in 0..n {
            for ch in 0..c {
                let j_start = ch.saturating_sub(half_size);
                let j_end = (ch + half_size_ceil).min(c - 1);

                for s in 0..spatial {
                    // Sum of squares over the window [j_start, j_end]
                    let mut sq_sum = 0.0f32;
                    for j in j_start..=j_end {
                        let v = inp[(batch * c + j) * spatial + s];
                        sq_sum += v * v;
                    }

                    let x = inp[(batch * c + ch) * spatial + s];
                    let norm = (self.bias + self.alpha / self.size as f32 * sq_sum).powf(self.beta);
                    buf[(batch * c + ch) * spatial + s] = x / norm;
                }
            }
        }

        output.set_dims(dims);
        Ok(())
    }
}
