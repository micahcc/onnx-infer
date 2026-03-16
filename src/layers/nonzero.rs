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

        // Collect indices of non-zero elements
        let mut coords: Vec<Vec<i64>> = vec![Vec::new(); rank];
        let numel = input.numel();

        let is_nonzero = |i: usize| -> bool {
            match input.dtype() {
                DType::Float => input.floats()[i] != 0.0,
                DType::Int64 => input.ints()[i] != 0,
            }
        };

        for flat in 0..numel {
            if is_nonzero(flat) {
                let mut remaining = flat;
                for ax in (0..rank).rev() {
                    coords[ax].push((remaining % input.dims[ax]) as i64);
                    remaining /= input.dims[ax];
                }
            }
        }

        let nnz = coords.first().map(|c| c.len()).unwrap_or(0);
        let total = rank * nnz;
        let buf = output.as_mut_i64(total);

        for (r, coord_row) in coords.iter().enumerate() {
            buf[r * nnz..(r + 1) * nnz].copy_from_slice(coord_row);
        }

        output.set_dims(&[rank, nnz]);
        Ok(())
    }
}
