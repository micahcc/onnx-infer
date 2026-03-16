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

        let in_rank = input.dims.len();
        let rank_i64 = in_rank as i64;

        // Build axes bitmask instead of HashSet
        let mut axes_mask = [false; 8];
        if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_t = get_tensor(values, &self.inputs[1])?;
            for &a in axes_t.ints() {
                let idx = if a < 0 {
                    (rank_i64 + a) as usize
                } else {
                    a as usize
                };
                axes_mask[idx] = true;
            }
        } else if let Some(ref attr) = self.axes_attr {
            for &a in attr {
                let idx = if a < 0 {
                    (rank_i64 + a) as usize
                } else {
                    a as usize
                };
                axes_mask[idx] = true;
            }
        } else {
            for i in 0..in_rank {
                axes_mask[i] = true;
            }
        }

        let mut out_dims = [0usize; 8];
        let mut out_rank = 0;
        for (i, &d) in input.dims.iter().enumerate() {
            if axes_mask[i] {
                if self.keepdims {
                    out_dims[out_rank] = 1;
                    out_rank += 1;
                }
            } else {
                out_dims[out_rank] = d;
                out_rank += 1;
            }
        }
        if out_rank == 0 {
            out_dims[0] = 1;
            out_rank = 1;
        }

        let out_numel: usize = out_dims[..out_rank].iter().product();
        let buf = output.as_mut_f32(out_numel);
        buf.fill(f32::INFINITY);

        let mut in_strides = [1usize; 8];
        for i in (0..in_rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        let mut out_strides = [1usize; 8];
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
                if !axes_mask[ax] {
                    out_flat += coord * out_strides[out_idx];
                    out_idx += 1;
                } else if self.keepdims {
                    out_idx += 1;
                }
            }
            buf[out_flat] = buf[out_flat].min(input_f[in_flat]);
        }

        output.set_dims(&out_dims[..out_rank]);
        Ok(())
    }
}
