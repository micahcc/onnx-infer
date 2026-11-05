use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::broadcast_index;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct GreaterOrEqual {
    pub inputs: Vec<String>,
}

impl GreaterOrEqual {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for GreaterOrEqual {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let ndim = a.dims.len().max(b.dims.len());
        let mut out_shape = [0usize; 8];
        broadcast_shape_into(&a.dims, &b.dims, &mut out_shape[..ndim]);
        let out_dims = &out_shape[..ndim];
        let numel: usize = out_dims.iter().product();

        if numel == 0 {
            output.as_mut_i64(0);
            output.set_dims(out_dims);
            return Ok(());
        }

        let buf = output.as_mut_i64(numel);
        let mut index = [0usize; 8];

        for val in buf.iter_mut() {
            let ai = broadcast_index(&index[..ndim], &a.dims, out_dims);
            let bi = broadcast_index(&index[..ndim], &b.dims, out_dims);
            let va: f64 = match a.dtype() {
                DType::Float => a.floats().context("GreaterOrEqual: a")?[ai] as f64,
                DType::Int64 => a.ints().context("GreaterOrEqual: a")?[ai] as f64,
                DType::String => unreachable!(),
            };
            let vb: f64 = match b.dtype() {
                DType::Float => b.floats().context("GreaterOrEqual: b")?[bi] as f64,
                DType::Int64 => b.ints().context("GreaterOrEqual: b")?[bi] as f64,
                DType::String => unreachable!(),
            };
            *val = if va >= vb { 1 } else { 0 };

            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < out_dims[d] {
                    break;
                }
                index[d] = 0;
            }
        }

        output.set_dims(out_dims);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constants;
    use crate::Values;
    use crate::dims;
    use crate::Tensor;
    use std::collections::HashMap;

    #[test]
    fn basic() {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("a".into(), Tensor::new(dims![4], vec![1.0f32, 2.0, 3.0, 4.0]));
        m.insert("b".into(), Tensor::new(dims![4], vec![2.0f32, 2.0, 2.0, 2.0]));
        let vals = Values { intermediates: &m, constants: Constants::empty() };
        let mut out = Tensor::new(dims![4], vec![0.0f32; 4]);
        let mut op = GreaterOrEqual::new(vec!["a".into(), "b".into()]);
        op.execute(&vals, &mut out).unwrap();
        let r = out.ints().unwrap();
        assert_eq!(r, &[0, 1, 1, 1]);
    }
}
