use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;

pub struct Split {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub axis: i64,
    pub split_sizes: [usize; 8],
    pub split_sizes_len: usize,
}

impl Split {
    pub fn new(
        inputs: Vec<String>,
        outputs: Vec<String>,
        axis: i64,
        split_sizes: Vec<i64>,
    ) -> Self {
        let mut sizes = [0usize; 8];
        for (i, &s) in split_sizes.iter().enumerate() {
            sizes[i] = s as usize;
        }
        Self {
            inputs,
            outputs,
            axis,
            split_sizes: sizes,
            split_sizes_len: split_sizes.len(),
        }
    }

    pub fn execute(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        let outer: usize = input.dims[..axis].iter().product();
        let inner: usize = input.dims[axis + 1..].iter().product();
        let in_rank = input.dims.len();
        let mut in_dims = [0usize; 8];
        in_dims[..in_rank].copy_from_slice(&input.dims);
        let dtype = input.dtype();

        // Compute split sizes
        let num_outputs = self.outputs.len();
        let mut sizes = [0usize; 8];
        if self.split_sizes_len > 0 {
            sizes[..num_outputs].copy_from_slice(&self.split_sizes[..num_outputs]);
        } else {
            let dim = in_dims[axis];
            let base = dim / num_outputs;
            let remainder = dim % num_outputs;
            for (i, s) in sizes.iter_mut().take(num_outputs).enumerate() {
                *s = base + if i < remainder { 1 } else { 0 };
            }
        }

        let mut offset = 0;
        for (i, out_name) in self.outputs.iter().enumerate() {
            if out_name.is_empty() {
                offset += sizes[i];
                continue;
            }
            let chunk = sizes[i];
            let numel = outer * chunk * inner;

            let mut out_dims = [0usize; 8];
            out_dims[..in_rank].copy_from_slice(&in_dims[..in_rank]);
            out_dims[axis] = chunk;
            let out_rank = in_rank;

            let (key, mut out) = values
                .remove_entry(out_name.as_str())
                .unwrap_or_else(|| (out_name.clone(), Tensor::default()));

            match dtype {
                DType::Float => {
                    let in_t = values.get(&self.inputs[0]).unwrap();
                    let in_data = in_t.floats();
                    let buf = out.as_mut_f32(numel);
                    let mut idx = 0;
                    for o in 0..outer {
                        let base = (o * in_dims[axis] + offset) * inner;
                        for c in 0..chunk {
                            let src = base + c * inner;
                            buf[idx..idx + inner].copy_from_slice(&in_data[src..src + inner]);
                            idx += inner;
                        }
                    }
                }
                DType::Int64 => {
                    let in_t = values.get(&self.inputs[0]).unwrap();
                    let in_data = in_t.ints();
                    let buf = out.as_mut_i64(numel);
                    let mut idx = 0;
                    for o in 0..outer {
                        let base = (o * in_dims[axis] + offset) * inner;
                        for c in 0..chunk {
                            let src = base + c * inner;
                            buf[idx..idx + inner].copy_from_slice(&in_data[src..src + inner]);
                            idx += inner;
                        }
                    }
                }
                DType::String => unreachable!("strings not supported"),
            }

            out.set_dims(&out_dims[..out_rank]);
            values.insert(key, out);
            offset += chunk;
        }

        Ok(())
    }
}
