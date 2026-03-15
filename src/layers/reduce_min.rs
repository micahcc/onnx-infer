use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::layers::Layer;
use crate::onnx::NodeProto;

pub struct ReduceMin {
    pub inputs: Vec<String>,
    pub keepdims: bool,
    pub axes_attr: Option<Vec<i64>>,
}

impl ReduceMin {
    pub fn new(inputs: Vec<String>, keepdims: bool, node: &NodeProto) -> Self {
        Self {
            inputs,
            keepdims,
            axes_attr: get_attr_ints(node, "axes"),
        }
    }
}

impl Layer for ReduceMin {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let axes: Vec<i64> = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_t = get_tensor(values, &self.inputs[1])?;
            axes_t.ints().to_vec()
        } else {
            self.axes_attr
                .clone()
                .unwrap_or_else(|| (0..input.dims.len() as i64).collect())
        };

        let rank = input.dims.len() as i64;
        let axes_set: std::collections::HashSet<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (rank + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        let mut out_dims = Vec::new();
        for (i, &d) in input.dims.iter().enumerate() {
            if axes_set.contains(&i) {
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

        let out_numel: usize = out_dims.iter().product();
        let buf = output.as_mut_f32(out_numel);
        buf.fill(f32::INFINITY);

        let in_rank = input.dims.len();
        let mut in_strides = vec![1usize; in_rank];
        for i in (0..in_rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        let out_rank = out_dims.len();
        let mut out_strides = vec![1usize; out_rank];
        if out_rank > 1 {
            for i in (0..out_rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
            }
        }

        let input_f = input.floats();
        for in_flat in 0..input.numel() {
            let mut remaining = in_flat;
            let mut out_flat = 0;
            let mut out_idx = 0;
            for ax in 0..in_rank {
                let coord = remaining / in_strides[ax];
                remaining %= in_strides[ax];
                if !axes_set.contains(&ax) {
                    out_flat += coord * out_strides[out_idx];
                    out_idx += 1;
                } else if self.keepdims {
                    out_idx += 1;
                }
            }
            buf[out_flat] = buf[out_flat].min(input_f[in_flat]);
        }

        output.dims = out_dims;
        Ok(())
    }
}
