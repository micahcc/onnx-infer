use anyhow::Context;
use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct NonZero {
    pub inputs: Vec<String>,
}

impl NonZero {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for NonZero {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();
        let numel = input.numel();

        let is_nonzero = |i: usize| -> anyhow::Result<bool> {
            match input.dtype() {
                DType::Float => Ok(input.floats().context("in NonZero layer")?[i] != 0.0),
                DType::Int64 => Ok(input.ints().context("in NonZero layer")?[i] != 0),
                DType::String => anyhow::bail!("strings not supported in NonZero"),
            }
        };

        // Pass 1: count non-zero elements
        let mut nnz = 0usize;
        for flat in 0..numel {
            if is_nonzero(flat).context("in NonZero layer")? {
                nnz += 1;
            }
        }

        let total = rank * nnz;
        let buf = output.as_mut_i64(total);

        // Pass 2: fill coordinates directly into output buffer
        // Layout is [rank, nnz] — row r contains axis-r coords for all non-zero elements
        let mut col = 0;
        for flat in 0..numel {
            if is_nonzero(flat).context("in NonZero layer")? {
                let mut remaining = flat;
                for ax in (0..rank).rev() {
                    buf[ax * nnz + col] = (remaining % input.dims[ax]) as i64;
                    remaining /= input.dims[ax];
                }
                col += 1;
            }
        }

        output.set_dims(&[rank, nnz]);
        Ok(())
    }
}
