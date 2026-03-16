use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct ScatterElements {
    pub inputs: Vec<String>,
    pub axis: i64,
}

impl ScatterElements {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for ScatterElements {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let data = get_tensor(values, &self.inputs[0])?;
        let indices = get_tensor(values, &self.inputs[1])?;
        let updates = get_tensor(values, &self.inputs[2])?;

        let rank = data.dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let numel = data.numel();

        // Compute strides for data
        let mut strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * data.dims[i + 1];
        }

        let idx_vals: Vec<i64> = match indices.dtype() {
            DType::Int64 => indices.ints().to_vec(),
            DType::Float => indices.floats().iter().map(|&v| v as i64).collect(),
        };

        // Compute strides for indices
        let mut idx_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            idx_strides[i] = idx_strides[i + 1] * indices.dims[i + 1];
        }

        let idx_numel = indices.numel();

        match data.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                buf.copy_from_slice(data.floats());
                let upd = updates.floats();

                for flat in 0..idx_numel {
                    // Convert flat index to multi-dim coords in indices tensor
                    let mut remaining = flat;
                    let mut data_flat = 0;
                    for d in 0..rank {
                        let coord = remaining / idx_strides[d];
                        remaining %= idx_strides[d];
                        if d == axis {
                            let mut idx = idx_vals[flat];
                            if idx < 0 {
                                idx += data.dims[axis] as i64;
                            }
                            data_flat += idx as usize * strides[d];
                        } else {
                            data_flat += coord * strides[d];
                        }
                    }
                    buf[data_flat] = upd[flat];
                }
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(numel);
                buf.copy_from_slice(data.ints());
                let upd = updates.ints();

                for flat in 0..idx_numel {
                    let mut remaining = flat;
                    let mut data_flat = 0;
                    for d in 0..rank {
                        let coord = remaining / idx_strides[d];
                        remaining %= idx_strides[d];
                        if d == axis {
                            let mut idx = idx_vals[flat];
                            if idx < 0 {
                                idx += data.dims[axis] as i64;
                            }
                            data_flat += idx as usize * strides[d];
                        } else {
                            data_flat += coord * strides[d];
                        }
                    }
                    buf[data_flat] = upd[flat];
                }
            }
        }

        output.set_dims(&data.dims);
        Ok(())
    }
}
