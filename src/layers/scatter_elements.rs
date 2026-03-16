use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct ScatterPrecomp {
    pub axis: usize,
    pub rank: usize,
    pub strides: [usize; 8],
    pub idx_strides: [usize; 8],
}

pub struct ScatterElements {
    pub inputs: Vec<String>,
    pub axis: i64,
    pub shape_cache: Dims,
    pub precomp: Option<ScatterPrecomp>,
}

impl ScatterElements {
    pub fn compute_shapes(
        axis: i64,
        data_shape: &[usize],
        indices_shape: &[usize],
    ) -> ScatterPrecomp {
        let rank = data_shape.len();
        let axis = if axis < 0 {
            (rank as i64 + axis) as usize
        } else {
            axis as usize
        };
        let mut strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * data_shape[i + 1];
        }
        let mut idx_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            idx_strides[i] = idx_strides[i + 1] * indices_shape[i + 1];
        }
        ScatterPrecomp {
            axis,
            rank,
            strides,
            idx_strides,
        }
    }

    pub fn new(
        inputs: Vec<String>,
        axis: i64,
        initial_data_shape: &[usize],
        initial_indices_shape: &[usize],
    ) -> Self {
        let (shape_cache, precomp) =
            if !initial_data_shape.is_empty() && !initial_indices_shape.is_empty() {
                let mut cache = Dims::from_slice(initial_data_shape);
                cache.extend_from_slice(initial_indices_shape);
                (
                    cache,
                    Some(Self::compute_shapes(
                        axis,
                        initial_data_shape,
                        initial_indices_shape,
                    )),
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

impl Layer for ScatterElements {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let data = get_tensor(values, &self.inputs[0])?;
        let indices = get_tensor(values, &self.inputs[1])?;
        let updates = get_tensor(values, &self.inputs[2])?;

        let mut key = Dims::from_slice(&data.dims);
        key.extend_from_slice(&indices.dims);
        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == key.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(self.axis, &data.dims, &indices.dims));
                self.shape_cache = key;
                self.precomp.as_ref().expect("just set")
            }
        };

        let numel = data.numel();
        let idx_numel = indices.numel();
        let idx_is_int = indices.dtype() == DType::Int64;

        let resolve_idx = |flat: usize| -> i64 {
            if idx_is_int {
                indices.ints()[flat]
            } else {
                indices.floats()[flat] as i64
            }
        };

        #[allow(clippy::needless_range_loop)]
        match data.dtype() {
            DType::Float => {
                let buf = output.as_mut_f32(numel);
                buf.copy_from_slice(data.floats());
                let upd = updates.floats();

                for flat in 0..idx_numel {
                    let mut remaining = flat;
                    let mut data_flat = 0;
                    for d in 0..p.rank {
                        let coord = remaining / p.idx_strides[d];
                        remaining %= p.idx_strides[d];
                        if d == p.axis {
                            let mut idx = resolve_idx(flat);
                            if idx < 0 {
                                idx += data.dims[p.axis] as i64;
                            }
                            data_flat += idx as usize * p.strides[d];
                        } else {
                            data_flat += coord * p.strides[d];
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
                    for d in 0..p.rank {
                        let coord = remaining / p.idx_strides[d];
                        remaining %= p.idx_strides[d];
                        if d == p.axis {
                            let mut idx = resolve_idx(flat);
                            if idx < 0 {
                                idx += data.dims[p.axis] as i64;
                            }
                            data_flat += idx as usize * p.strides[d];
                        } else {
                            data_flat += coord * p.strides[d];
                        }
                    }
                    buf[data_flat] = upd[flat];
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        output.set_dims(&data.dims);
        Ok(())
    }
}
