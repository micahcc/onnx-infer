use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Transpose {
    pub inputs: Vec<String>,
    pub perm: Option<Vec<usize>>,
    scratch_perm: [usize; 8],
    scratch_in_strides: [usize; 8],
    scratch_out_strides: [usize; 8],
}

impl Transpose {
    pub fn new(inputs: Vec<String>, perm: Option<Vec<usize>>) -> Self {
        let mut scratch_perm = [0usize; 8];
        if let Some(ref p) = perm {
            for (i, &v) in p.iter().enumerate() {
                scratch_perm[i] = v;
            }
        }
        Self {
            inputs,
            perm,
            scratch_perm,
            scratch_in_strides: [0; 8],
            scratch_out_strides: [0; 8],
        }
    }
}

impl Layer for Transpose {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();

        if self.perm.is_none() {
            for i in 0..rank {
                self.scratch_perm[i] = rank - 1 - i;
            }
        }
        let perm = &self.scratch_perm;

        let mut out_dims = [0usize; 8];
        for i in 0..rank {
            out_dims[i] = input.dims[perm[i]];
        }

        self.scratch_in_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            self.scratch_in_strides[i] = self.scratch_in_strides[i + 1] * input.dims[i + 1];
        }
        let in_strides = &self.scratch_in_strides;

        let numel = input.numel();
        self.scratch_out_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            self.scratch_out_strides[i] = self.scratch_out_strides[i + 1] * out_dims[i + 1];
        }
        let out_strides = &self.scratch_out_strides;

        #[allow(clippy::needless_range_loop)]
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
        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
