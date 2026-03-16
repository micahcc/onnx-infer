use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;

pub struct TopK {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub axis: i64,
    pub largest: bool,
    in_buf: Vec<f32>,
}

impl TopK {
    pub fn new(inputs: Vec<String>, outputs: Vec<String>, axis: i64, largest: bool) -> Self {
        Self {
            inputs,
            outputs,
            axis,
            largest,
            in_buf: Vec::new(),
        }
    }

    pub fn execute(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let k_tensor = get_tensor(values, &self.inputs[1])?;
        let k = k_tensor.i64_at(0) as usize;

        let rank = input.dims.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        let outer: usize = input.dims[..axis].iter().product();
        let axis_size = input.dims[axis];
        let inner: usize = input.dims[axis + 1..].iter().product();
        let in_rank = input.dims.len();
        let mut in_dims = [0usize; 8];
        in_dims[..in_rank].copy_from_slice(&input.dims);

        // Copy input data to local buffer to release borrow on values
        self.in_buf.clear();
        self.in_buf.extend_from_slice(input.floats());

        let mut out_dims = [0usize; 8];
        out_dims[..in_rank].copy_from_slice(&in_dims[..in_rank]);
        out_dims[axis] = k;
        let out_rank = in_rank;
        let numel = outer * k * inner;

        // Values output
        let val_name = &self.outputs[0];
        let (val_key, mut val_out) = values
            .remove_entry(val_name.as_str())
            .unwrap_or_else(|| (val_name.clone(), Tensor::default()));
        let val_buf = val_out.as_mut_f32(numel);

        // Indices output
        let idx_name = &self.outputs[1];
        let (idx_key, mut idx_out) = values
            .remove_entry(idx_name.as_str())
            .unwrap_or_else(|| (idx_name.clone(), Tensor::default()));
        let idx_buf = idx_out.as_mut_i64(numel);

        let in_data = &self.in_buf;
        let mut indices: Vec<usize> = Vec::with_capacity(axis_size);
        let largest = self.largest;

        for o in 0..outer {
            for i in 0..inner {
                indices.clear();
                indices.extend(0..axis_size);
                if largest {
                    indices.sort_by(|&a, &b| {
                        let va = in_data[o * axis_size * inner + a * inner + i];
                        let vb = in_data[o * axis_size * inner + b * inner + i];
                        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    indices.sort_by(|&a, &b| {
                        let va = in_data[o * axis_size * inner + a * inner + i];
                        let vb = in_data[o * axis_size * inner + b * inner + i];
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                for (j, &src_idx) in indices.iter().take(k).enumerate() {
                    let dst = o * k * inner + j * inner + i;
                    val_buf[dst] = in_data[o * axis_size * inner + src_idx * inner + i];
                    idx_buf[dst] = src_idx as i64;
                }
            }
        }

        val_out.set_dims(&out_dims[..out_rank]);
        idx_out.set_dims(&out_dims[..out_rank]);
        values.insert(val_key, val_out);
        values.insert(idx_key, idx_out);

        Ok(())
    }
}
