use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

/// ReduceL2: `sqrt(sum(x^2, axes))`.
#[derive(Debug)]
pub struct ReduceL2 {
    inputs: Vec<String>,
    keepdims: bool,
    axes_attr: Option<Vec<i64>>,
    noop_with_empty_axes: bool,
}

impl ReduceL2 {
    pub fn new(
        inputs: Vec<String>,
        keepdims: bool,
        axes_attr: Option<Vec<i64>>,
        noop_with_empty_axes: bool,
    ) -> Self {
        Self { inputs, keepdims, axes_attr, noop_with_empty_axes }
    }
}

impl Layer for ReduceL2 {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let in_rank = input.dims.len();
        let rank_i64 = in_rank as i64;

        let axes_from_input: Option<Vec<i64>> =
            if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
                let t = get_tensor(values, &self.inputs[1])?;
                Some(t.ints().context("ReduceL2: axes")?.to_vec())
            } else {
                None
            };

        let axes_vec = axes_from_input.as_ref().or(self.axes_attr.as_ref());

        let mut axes_mask = [false; 8];
        match axes_vec {
            Some(axes) if axes.is_empty() && self.noop_with_empty_axes => {
                output.copy_from(input);
                return Ok(());
            }
            Some(axes) if axes.is_empty() => {
                for m in axes_mask.iter_mut().take(in_rank) { *m = true; }
            }
            Some(axes) => {
                for &a in axes {
                    let idx = if a < 0 { (rank_i64 + a) as usize } else { a as usize };
                    axes_mask[idx] = true;
                }
            }
            None => {
                for m in axes_mask.iter_mut().take(in_rank) { *m = true; }
            }
        }

        let mut out_dims = [0usize; 8];
        let mut out_rank = 0;
        for (i, &d) in input.dims.iter().enumerate() {
            if axes_mask[i] {
                if self.keepdims { out_dims[out_rank] = 1; out_rank += 1; }
            } else {
                out_dims[out_rank] = d; out_rank += 1;
            }
        }
        if out_rank == 0 { out_dims[0] = 1; out_rank = 1; }

        let out_numel: usize = out_dims[..out_rank].iter().product();

        let mut in_strides = [1usize; 8];
        for i in (0..in_rank.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }
        let mut out_strides = [1usize; 8];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }

        let keepdims = self.keepdims;
        let calc_out_flat = |in_flat: usize| -> usize {
            let mut rem = in_flat;
            let mut out_flat = 0;
            let mut out_idx = 0;
            for ax in 0..in_rank {
                let coord = rem / in_strides[ax];
                rem %= in_strides[ax];
                if !axes_mask[ax] {
                    out_flat += coord * out_strides[out_idx];
                    out_idx += 1;
                } else if keepdims {
                    out_idx += 1;
                }
            }
            out_flat
        };

        let buf = output.as_mut_f32(out_numel);
        buf.fill(0.0);
        let input_f = input.floats().context("ReduceL2: input")?;
        for (in_flat, &val) in input_f.iter().enumerate() {
            let of = calc_out_flat(in_flat);
            buf[of] += val * val;
        }
        for v in buf.iter_mut() {
            *v = v.sqrt();
        }

        output.set_dims(&out_dims[..out_rank]);
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
    fn reduce_all() {
        // sqrt(1+4+9+16) = sqrt(30)
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("x".into(), Tensor::new(dims![4], vec![1.0f32, 2.0, 3.0, 4.0]));
        let vals = Values { intermediates: &m, constants: Constants::empty() };
        let mut out = Tensor::new(dims![1], vec![0.0f32]);
        let mut op = ReduceL2::new(vec!["x".into()], false, Some(vec![]), false);
        op.execute(&vals, &mut out).unwrap();
        let expected = (1.0f32 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((out.floats().unwrap()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn reduce_axis0_keepdims() {
        // shape [2,3], reduce axis=0 keepdims=true → shape [1,3]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("x".into(), Tensor::new(dims![2, 3], data));
        let vals = Values { intermediates: &m, constants: Constants::empty() };
        let mut out = Tensor::new(dims![1, 3], vec![0.0f32; 3]);
        let mut op = ReduceL2::new(vec!["x".into()], true, Some(vec![0]), false);
        op.execute(&vals, &mut out).unwrap();
        let r = out.floats().unwrap();
        // col0: sqrt(1+16)=sqrt(17), col1: sqrt(4+25)=sqrt(29), col2: sqrt(9+36)=sqrt(45)
        assert!((r[0] - 17.0f32.sqrt()).abs() < 1e-5);
        assert!((r[1] - 29.0f32.sqrt()).abs() < 1e-5);
        assert!((r[2] - 45.0f32.sqrt()).abs() < 1e-5);
    }
}
