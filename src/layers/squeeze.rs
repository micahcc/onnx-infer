use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Squeeze {
    pub inputs: Vec<String>,
    pub axes_attr: Vec<i64>,
}

impl Squeeze {
    pub fn new(inputs: Vec<String>, axes_attr: Vec<i64>) -> Self {
        Self { inputs, axes_attr }
    }
}

impl Layer for Squeeze {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let axes: &[i64] = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_tensor = get_tensor(values, &self.inputs[1])?;
            axes_tensor.ints().context("in Squeeze layer")?
        } else {
            &self.axes_attr
        };

        let rank = input.dims.len();
        let mut squeeze_mask = [false; 8];
        if axes.is_empty() {
            for (i, &d) in input.dims.iter().enumerate() {
                if d == 1 {
                    squeeze_mask[i] = true;
                }
            }
        } else {
            for &a in axes {
                let idx = if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                };
                squeeze_mask[idx] = true;
            }
        }

        let mut out_dims = [0usize; 8];
        let mut out_rank = 0;
        for (i, &d) in input.dims.iter().enumerate() {
            if !squeeze_mask[i] {
                out_dims[out_rank] = d;
                out_rank += 1;
            }
        }

        output.copy_from(input);
        output.set_dims(&out_dims[..out_rank]);
        Ok(())
    }
}
