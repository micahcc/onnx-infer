use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct ArgMax {
    pub inputs: Vec<String>,
    pub axis: i64,
    pub keepdims: bool,
    pub select_last_index: bool,
}

impl ArgMax {
    pub fn new(inputs: Vec<String>, axis: i64, keepdims: bool, select_last_index: bool) -> Self {
        Self {
            inputs,
            axis,
            keepdims,
            select_last_index,
        }
    }
}

impl Layer for ArgMax {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let data = input.floats().context("in ArgMax layer")?;
        let axis_len = input.dims[axis];

        // Compute outer and inner strides
        let outer: usize = input.dims[..axis].iter().product();
        let inner: usize = input.dims[axis + 1..].iter().product();

        let out_numel = outer * inner;
        let buf = output.as_mut_i64(out_numel);

        for o in 0..outer {
            for i in 0..inner {
                let mut best_idx = 0i64;
                let mut best_val = f32::NEG_INFINITY;
                for a in 0..axis_len {
                    let flat = o * axis_len * inner + a * inner + i;
                    let v = data[flat];
                    if v > best_val || (self.select_last_index && v >= best_val) {
                        best_val = v;
                        best_idx = a as i64;
                    }
                }
                buf[o * inner + i] = best_idx;
            }
        }

        let mut out_dims = Vec::with_capacity(rank);
        for (i, &d) in input.dims.iter().enumerate() {
            if i == axis {
                if self.keepdims {
                    out_dims.push(1);
                }
            } else {
                out_dims.push(d);
            }
        }
        if out_dims.is_empty() {
            out_dims.push(1);
        }
        output.set_dims(&out_dims);
        Ok(())
    }
}
