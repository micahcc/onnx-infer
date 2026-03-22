use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct ReduceMean {
    pub inputs: Vec<String>,
    pub keepdims: bool,
    pub axes_attr: Option<Vec<i64>>,
    pub noop_with_empty_axes: bool,
}

impl ReduceMean {
    pub fn new(
        inputs: Vec<String>,
        keepdims: bool,
        axes_attr: Option<Vec<i64>>,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self {
            inputs,
            keepdims,
            axes_attr,
            noop_with_empty_axes,
        }
    }
}

impl Layer for ReduceMean {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let in_rank = input.dims.len();
        let rank_i64 = in_rank as i64;

        let axes_from_input = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_t = get_tensor(values, &self.inputs[1])?;
            Some(match axes_t.dtype() {
                DType::Int64 => axes_t.ints().context("in ReduceMean layer")?.to_vec(),
                DType::Float => axes_t
                    .floats()
                    .context("in ReduceMean layer")?
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                DType::String => unreachable!("strings not supported"),
            })
        } else {
            None
        };

        let axes_vec = axes_from_input.as_ref().or(self.axes_attr.as_ref());

        let axes_mask = if let Some(axes) = axes_vec {
            if axes.is_empty() && self.noop_with_empty_axes {
                output.copy_from(input);
                return Ok(());
            }
            if axes.is_empty() {
                let mut mask = [false; 8];
                for m in mask.iter_mut().take(in_rank) {
                    *m = true;
                }
                mask
            } else {
                let mut mask = [false; 8];
                for &a in axes {
                    let idx = if a < 0 {
                        (rank_i64 + a) as usize
                    } else {
                        a as usize
                    };
                    mask[idx] = true;
                }
                mask
            }
        } else {
            // No axes specified: reduce all
            let mut mask = [false; 8];
            for m in mask.iter_mut().take(in_rank) {
                *m = true;
            }
            mask
        };

        // Compute the number of elements being averaged over
        let reduction_count: usize = input
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| axes_mask[*i])
            .map(|(_, &d)| d)
            .product();

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

        let mut in_strides = [1usize; 8];
        if in_rank > 1 {
            for i in (0..in_rank - 1).rev() {
                in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
            }
        }

        let mut out_strides = [1usize; 8];
        if out_rank > 1 {
            for i in (0..out_rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
            }
        }

        let keepdims = self.keepdims;
        let calc_out_flat = |in_flat: usize| -> usize {
            let mut remaining = in_flat;
            let mut out_flat = 0;
            let mut out_idx = 0;
            for ax in 0..in_rank {
                let coord = remaining / in_strides[ax];
                remaining %= in_strides[ax];
                if !axes_mask[ax] {
                    out_flat += coord * out_strides[out_idx];
                    out_idx += 1;
                } else if keepdims {
                    out_idx += 1;
                }
            }
            out_flat
        };

        // Only float is meaningful for mean; int64 mean rounds to float then back
        match input.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(out_numel);
                buf.fill(0.0);
                let input_f = input.floats().context("in ReduceMean layer")?;
                for (in_flat, &val) in input_f.iter().enumerate() {
                    let of = calc_out_flat(in_flat);
                    buf[of] += val;
                }
                if reduction_count > 1 {
                    let inv = 1.0 / reduction_count as f32;
                    for v in buf.iter_mut() {
                        *v *= inv;
                    }
                }
            }
            DType::Int64 => {
                // Produce float output for int64 mean (consistent with ONNX spec)
                let buf = output.as_mut_f32(out_numel);
                buf.fill(0.0);
                let input_i = input.ints().context("in ReduceMean layer")?;
                for (in_flat, &val) in input_i.iter().enumerate() {
                    let of = calc_out_flat(in_flat);
                    buf[of] += val as f32;
                }
                if reduction_count > 1 {
                    let inv = 1.0 / reduction_count as f32;
                    for v in buf.iter_mut() {
                        *v *= inv;
                    }
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        output.set_dims(&out_dims[..out_rank]);
        Ok(())
    }
}
