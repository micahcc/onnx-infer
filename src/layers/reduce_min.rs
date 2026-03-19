use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct ReduceMinPrecomp {
    pub axes_mask: [bool; 8],
    pub out_dims: [usize; 8],
    pub out_rank: usize,
    pub out_numel: usize,
    pub in_strides: [usize; 8],
    pub out_strides: [usize; 8],
    pub in_rank: usize,
}

pub struct ReduceMin {
    pub inputs: Vec<String>,
    pub keepdims: bool,
    pub axes_attr_mask: Option<[bool; 8]>,
    pub axes_attr_raw: Option<Vec<i64>>,
    pub shape_cache: Dims,
    pub precomp: Option<ReduceMinPrecomp>,
}

impl ReduceMin {
    pub fn compute_shapes(
        shape: &[usize],
        axes_mask: &[bool; 8],
        keepdims: bool,
    ) -> ReduceMinPrecomp {
        let in_rank = shape.len();
        let mut out_dims = [0usize; 8];
        let mut out_rank = 0;
        for (i, &d) in shape.iter().enumerate() {
            if axes_mask[i] {
                if keepdims {
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
        for i in (0..in_rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1];
        }

        let mut out_strides = [1usize; 8];
        if out_rank > 1 {
            for i in (0..out_rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
            }
        }

        ReduceMinPrecomp {
            axes_mask: *axes_mask,
            out_dims,
            out_rank,
            out_numel,
            in_strides,
            out_strides,
            in_rank,
        }
    }

    pub fn new(
        inputs: Vec<String>,
        keepdims: bool,
        axes_attr: Option<Vec<i64>>,
        initial_shape: &[usize],
    ) -> Self {
        let has_negative = axes_attr.as_ref().is_some_and(|a| a.iter().any(|&v| v < 0));
        let axes_attr_mask = if !has_negative {
            axes_attr.as_ref().map(|a| {
                let mut mask = [false; 8];
                for &v in a {
                    mask[v as usize] = true;
                }
                mask
            })
        } else {
            None
        };

        let (shape_cache, precomp) = if !initial_shape.is_empty() {
            if let Some(mask) = axes_attr_mask {
                (
                    Dims::from_slice(initial_shape),
                    Some(Self::compute_shapes(initial_shape, &mask, keepdims)),
                )
            } else {
                (Dims::from_slice(initial_shape), None)
            }
        } else {
            (Dims::new(), None)
        };

        Self {
            inputs,
            keepdims,
            axes_attr_mask,
            axes_attr_raw: if has_negative { axes_attr } else { None },
            shape_cache,
            precomp,
        }
    }
}

impl Layer for ReduceMin {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        // Try to use precomp fast path (static axes mask known at build time)
        if let Some(mask) = self.axes_attr_mask {
            let p = match &self.precomp {
                Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
                _ => {
                    self.precomp = Some(Self::compute_shapes(&input.dims, &mask, self.keepdims));
                    self.shape_cache.clone_from(&input.dims);
                    self.precomp.as_ref().expect("just set")
                }
            };

            let axes_mask = &p.axes_mask;
            let in_strides = &p.in_strides;
            let out_strides = &p.out_strides;
            let in_rank = p.in_rank;
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

            match input.dtype() {
                DType::Float => {
                    let buf = output.as_mut_f32(p.out_numel);
                    buf.fill(f32::INFINITY);
                    let input_f = input.floats();
                    for (in_flat, &val) in input_f.iter().enumerate() {
                        let of = calc_out_flat(in_flat);
                        buf[of] = buf[of].min(val);
                    }
                }
                DType::Int64 => {
                    let buf = output.as_mut_i64(p.out_numel);
                    buf.fill(i64::MAX);
                    let input_i = input.ints();
                    for (in_flat, &val) in input_i.iter().enumerate() {
                        let of = calc_out_flat(in_flat);
                        buf[of] = buf[of].min(val);
                    }
                }
                DType::String => unreachable!("strings not supported"),
            }

            output.set_dims(&p.out_dims[..p.out_rank]);
            return Ok(());
        }

        // Slow path: compute axes mask at runtime
        let in_rank = input.dims.len();
        let rank_i64 = in_rank as i64;

        let axes_mask = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_t = get_tensor(values, &self.inputs[1])?;
            let mut mask = [false; 8];
            for &a in axes_t.ints() {
                let idx = if a < 0 {
                    (rank_i64 + a) as usize
                } else {
                    a as usize
                };
                mask[idx] = true;
            }
            mask
        } else if let Some(ref attr) = self.axes_attr_raw {
            let mut mask = [false; 8];
            for &a in attr {
                let idx = (rank_i64 + a) as usize;
                mask[idx] = true;
            }
            mask
        } else {
            let mut mask = [false; 8];
            for m in mask.iter_mut().take(in_rank) {
                *m = true;
            }
            mask
        };

        let p = Self::compute_shapes(&input.dims, &axes_mask, self.keepdims);

        let keepdims = self.keepdims;
        #[allow(clippy::needless_range_loop)]
        let calc_out_flat = |in_flat: usize| -> usize {
            let mut remaining = in_flat;
            let mut out_flat = 0;
            let mut out_idx = 0;
            for ax in 0..p.in_rank {
                let coord = remaining / p.in_strides[ax];
                remaining %= p.in_strides[ax];
                if !axes_mask[ax] {
                    out_flat += coord * p.out_strides[out_idx];
                    out_idx += 1;
                } else if keepdims {
                    out_idx += 1;
                }
            }
            out_flat
        };

        match input.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(p.out_numel);
                buf.fill(f32::INFINITY);
                let input_f = input.floats();
                for (in_flat, &val) in input_f.iter().enumerate() {
                    let of = calc_out_flat(in_flat);
                    buf[of] = buf[of].min(val);
                }
            }
            DType::Int64 => {
                let buf = output.as_mut_i64(p.out_numel);
                buf.fill(i64::MAX);
                let input_i = input.ints();
                for (in_flat, &val) in input_i.iter().enumerate() {
                    let of = calc_out_flat(in_flat);
                    buf[of] = buf[of].min(val);
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        output.set_dims(&p.out_dims[..p.out_rank]);
        Ok(())
    }
}
