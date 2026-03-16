use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct TransposePrecomp {
    pub perm: [usize; 8],
    pub in_strides: [usize; 8],
    pub out_strides: [usize; 8],
    pub out_dims: [usize; 8],
    pub rank: usize,
    pub numel: usize,
}

pub struct Transpose {
    pub inputs: Vec<String>,
    pub perm: Option<Vec<usize>>,
    pub shape_cache: Dims,
    pub precomp: Option<TransposePrecomp>,
}

impl Transpose {
    pub fn compute_shapes(perm: Option<&[usize]>, shape: &[usize]) -> TransposePrecomp {
        let rank = shape.len();
        let mut scratch_perm = [0usize; 8];
        if let Some(p) = perm {
            for (i, &v) in p.iter().enumerate() {
                scratch_perm[i] = v;
            }
        } else {
            for (i, slot) in scratch_perm.iter_mut().enumerate().take(rank) {
                *slot = rank - 1 - i;
            }
        }

        let mut out_dims = [0usize; 8];
        for i in 0..rank {
            out_dims[i] = shape[scratch_perm[i]];
        }

        let mut in_strides = [0usize; 8];
        in_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1];
        }

        let mut out_strides = [0usize; 8];
        out_strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }

        TransposePrecomp {
            perm: scratch_perm,
            in_strides,
            out_strides,
            out_dims,
            rank,
            numel: shape.iter().product(),
        }
    }

    pub fn new(inputs: Vec<String>, perm: Option<Vec<usize>>, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if !initial_shape.is_empty() {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(perm.as_deref(), initial_shape)),
            )
        } else {
            (Dims::new(), None)
        };

        Self {
            inputs,
            perm,
            shape_cache,
            precomp,
        }
    }
}

impl Layer for Transpose {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(self.perm.as_deref(), &input.dims));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };

        let perm = &p.perm;
        let in_strides = &p.in_strides;
        let out_strides = &p.out_strides;
        let rank = p.rank;
        let numel = p.numel;

        #[allow(clippy::needless_range_loop)]
        match input.dtype() {
            DType::Float => {
                let in_data = input.floats();
                let buf = output.as_mut_f32(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for i in 0..rank {
                        let coord = remaining / out_strides[i];
                        remaining %= out_strides[i];
                        in_flat += coord * in_strides[perm[i]];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
            DType::Int64 => {
                let in_data = input.ints();
                let buf = output.as_mut_i64(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for i in 0..rank {
                        let coord = remaining / out_strides[i];
                        remaining %= out_strides[i];
                        in_flat += coord * in_strides[perm[i]];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
            DType::String => unreachable!("strings not supported"),
        }
        output.set_dims(&p.out_dims[..rank]);
        Ok(())
    }
}
