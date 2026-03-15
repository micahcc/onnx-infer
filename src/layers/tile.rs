use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Tile {
    pub inputs: Vec<String>,
}

impl Tile {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Tile {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let repeats_t = get_tensor(values, &self.inputs[1])?;
        let repeats: Vec<usize> = repeats_t.ints().iter().map(|&v| v as usize).collect();

        let rank = input.dims.len();
        let mut out_dims = vec![0usize; rank];
        for i in 0..rank {
            out_dims[i] = input.dims[i] * repeats[i];
        }

        let numel: usize = out_dims.iter().product();

        let mut out_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }
        let mut in_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        match input.dtype {
            DType::Float => {
                let in_data = input.floats();
                let buf = output.as_mut_f32(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for ax in 0..rank {
                        let coord = remaining / out_strides[ax];
                        remaining %= out_strides[ax];
                        in_flat += (coord % input.dims[ax]) * in_strides[ax];
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
                    for ax in 0..rank {
                        let coord = remaining / out_strides[ax];
                        remaining %= out_strides[ax];
                        in_flat += (coord % input.dims[ax]) * in_strides[ax];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
        }
        output.dims = out_dims;
        Ok(())
    }
}
