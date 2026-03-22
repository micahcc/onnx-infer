use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct TransposePrecomp {
    pub perm: [usize; 8],
    pub in_strides: [usize; 8],
    pub out_dims: [usize; 8],
    pub rank: usize,
    pub numel: usize,
    /// Precomputed: in_strides[perm[i]] for each output dimension
    pub perm_in_strides: [usize; 8],
}

#[derive(Debug)]
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

        let mut perm_in_strides = [0usize; 8];
        for i in 0..rank {
            perm_in_strides[i] = in_strides[scratch_perm[i]];
        }

        TransposePrecomp {
            perm: scratch_perm,
            in_strides,
            out_dims,
            rank,
            numel: shape.iter().product(),
            perm_in_strides,
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

fn transpose_inner<T: Copy>(
    src: &[T],
    dst: &mut [T],
    out_dims: &[usize; 8],
    perm_in_strides: &[usize; 8],
    rank: usize,
    numel: usize,
) {
    // Use an incrementing coordinate array to avoid division/modulo
    let mut coord = [0usize; 8];
    let mut in_off = 0usize;
    for (_out_flat, dst_value) in dst.iter_mut().enumerate().take(numel) {
        *dst_value = src[in_off];

        // Increment coordinate and update in_off
        for d in (0..rank).rev() {
            coord[d] += 1;
            in_off += perm_in_strides[d];
            if coord[d] < out_dims[d] {
                break;
            }
            in_off -= coord[d] * perm_in_strides[d];
            coord[d] = 0;
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

        let rank = p.rank;
        let numel = p.numel;

        match input.dtype() {
            DType::Float => {
                let in_data = input.floats().context("in Transpose layer")?;
                let buf = output.as_mut_f32(numel);
                transpose_inner(in_data, buf, &p.out_dims, &p.perm_in_strides, rank, numel);
            }
            DType::Int64 => {
                let in_data = input.ints().context("in Transpose layer")?;
                let buf = output.as_mut_i64(numel);
                transpose_inner(in_data, buf, &p.out_dims, &p.perm_in_strides, rank, numel);
            }
            DType::String => unreachable!("strings not supported"),
        }
        output.set_dims(&p.out_dims[..rank]);
        Ok(())
    }
}
