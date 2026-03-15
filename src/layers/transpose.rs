use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Transpose {
    pub inputs: Vec<String>,
    pub perm: Option<Vec<usize>>,
}

impl Transpose {
    pub fn new(inputs: Vec<String>, perm: Option<Vec<usize>>) -> Self {
        Self { inputs, perm }
    }
}

impl Layer for Transpose {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();

        let perm: Vec<usize> = self
            .perm
            .clone()
            .unwrap_or_else(|| (0..rank).rev().collect());

        let mut out_dims = vec![0usize; rank];
        for i in 0..rank {
            out_dims[i] = input.dims[perm[i]];
        }

        let mut in_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        let numel = input.numel();
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }

        match input.dtype() {
            DType::Float => {
                let in_data = input.floats();
                let buf = output.as_mut_f32(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for i in 0..rank {
                        let coord = remaining / out_strides[i];
                        remaining %= out_strides[i];
                        in_flat += coord * in_strides[perm[i]];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
            DType::Int64 => {
                let in_data = input.ints();
                let buf = output.as_mut_i64(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for i in 0..rank {
                        let coord = remaining / out_strides[i];
                        remaining %= out_strides[i];
                        in_flat += coord * in_strides[perm[i]];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
        }
        output.dims = out_dims;
        Ok(())
    }
}
