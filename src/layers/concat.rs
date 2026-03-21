use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct ConcatPrecomp {
    pub axis: usize,
    pub rank: usize,
    pub outer: usize,
    pub inner: usize,
}

pub struct Concat {
    pub inputs: Vec<String>,
    pub axis: i64,
    pub shape_cache: Dims,
    pub precomp: Option<ConcatPrecomp>,
}

impl Concat {
    pub fn compute_shapes(axis: i64, shape: &[usize]) -> ConcatPrecomp {
        let rank = shape.len();
        let axis = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };
        ConcatPrecomp {
            axis,
            rank,
            outer: shape[..axis].iter().product(),
            inner: shape[axis + 1..].iter().product(),
        }
    }

    pub fn new(inputs: Vec<String>, axis: i64, initial_shapes: &[&[usize]]) -> Self {
        let (shape_cache, precomp) =
            if let Some(shape) = initial_shapes.iter().find(|s| !s.is_empty()) {
                (
                    Dims::from_slice(shape),
                    Some(Self::compute_shapes(axis, shape)),
                )
            } else {
                (Dims::new(), None)
            };

        Self {
            inputs,
            axis,
            shape_cache,
            precomp,
        }
    }
}

impl Layer for Concat {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let first_name = self
            .inputs
            .iter()
            .find(|n| !n.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Concat with no inputs"))?;
        let first = get_tensor(values, first_name)?;

        let p = match &self.precomp {
            Some(p)
                if p.rank == first.dims.len() && {
                    let rt_outer: usize = first.dims[..p.axis].iter().product();
                    let rt_inner: usize = first.dims[p.axis + 1..].iter().product();
                    rt_outer == p.outer && rt_inner == p.inner
                } =>
            {
                p
            }
            _ => {
                self.precomp = Some(Self::compute_shapes(self.axis, &first.dims));
                self.shape_cache.clone_from(&first.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let axis = p.axis;
        let rank = p.rank;
        let outer = p.outer;
        let inner = p.inner;

        let mut out_dims = [0usize; 8];
        for (i, &d) in first.dims.iter().enumerate() {
            out_dims[i] = d;
        }
        let mut axis_sum = 0usize;
        for name in &self.inputs {
            if !name.is_empty() {
                let t = get_tensor(values, name)?;
                axis_sum += t.dims[axis];
            }
        }
        out_dims[axis] = axis_sum;

        let dtype = first.dtype();

        match dtype {
            DType::Int64 => {
                let total = out_dims[..rank].iter().product::<usize>();
                let buf = output.as_mut_i64(total);
                let mut axis_offset = 0;
                for name in &self.inputs {
                    if name.is_empty() {
                        continue;
                    }
                    let t = get_tensor(values, name)?;
                    let t_axis = t.dims[axis];
                    if t.dtype() == DType::Float {
                        let floats = t.floats().context("in Concat layer")?;
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                for k in 0..inner {
                                    buf[dst_base + k] = floats[src_base + k] as i64;
                                }
                            }
                        }
                    } else {
                        let t_data = t.ints().context("in Concat layer")?;
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                buf[dst_base..dst_base + inner]
                                    .copy_from_slice(&t_data[src_base..src_base + inner]);
                            }
                        }
                    }
                    axis_offset += t_axis;
                }
            }
            DType::Float => {
                let total = out_dims[..rank].iter().product::<usize>();
                let buf = output.as_mut_f32(total);
                let mut axis_offset = 0;
                for name in &self.inputs {
                    if name.is_empty() {
                        continue;
                    }
                    let t = get_tensor(values, name)?;
                    let t_axis = t.dims[axis];
                    if t.dtype() == DType::Int64 {
                        let ints = t.ints().context("in Concat layer")?;
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                for k in 0..inner {
                                    buf[dst_base + k] = ints[src_base + k] as f32;
                                }
                            }
                        }
                    } else {
                        let t_data = t.floats().context("in Concat layer")?;
                        for o in 0..outer {
                            for a in 0..t_axis {
                                let src_base = (o * t_axis + a) * inner;
                                let dst_base = (o * out_dims[axis] + axis_offset + a) * inner;
                                buf[dst_base..dst_base + inner]
                                    .copy_from_slice(&t_data[src_base..src_base + inner]);
                            }
                        }
                    }
                    axis_offset += t_axis;
                }
            }
            DType::String => unreachable!("strings not supported"),
        }
        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
