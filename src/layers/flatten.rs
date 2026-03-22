use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct FlattenPrecomp {
    pub outer: usize,
    pub inner: usize,
}

#[derive(Debug)]
pub struct Flatten {
    pub inputs: Vec<String>,
    pub axis: usize,
    pub shape_cache: Dims,
    pub precomp: Option<FlattenPrecomp>,
}

impl Flatten {
    pub fn compute_shapes(axis: usize, shape: &[usize]) -> FlattenPrecomp {
        FlattenPrecomp {
            outer: shape[..axis].iter().product(),
            inner: shape[axis..].iter().product(),
        }
    }

    pub fn new(inputs: Vec<String>, axis: usize, initial_shape: &[usize]) -> Self {
        let (shape_cache, precomp) = if !initial_shape.is_empty() {
            (
                Dims::from_slice(initial_shape),
                Some(Self::compute_shapes(axis, initial_shape)),
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

impl Layer for Flatten {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let p = match &self.precomp {
            Some(p) if self.shape_cache.as_slice() == input.dims.as_slice() => p,
            _ => {
                self.precomp = Some(Self::compute_shapes(self.axis, &input.dims));
                self.shape_cache.clone_from(&input.dims);
                self.precomp.as_ref().expect("just set")
            }
        };
        output.copy_from(input);
        output.set_dims(&[p.outer, p.inner]);
        Ok(())
    }
}
