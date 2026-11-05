//! LayerNormalization operator.
//!
//! ONNX spec: <https://onnx.ai/onnx/operators/onnx__LayerNormalization.html>
//!
//! Normalises over all axes from `axis` to the end of the tensor, then
//! applies a learned `scale` and optional `bias`.  The common case is
//! `axis = -1` (or equivalently `rank - 1`): one vector per spatial/token
//! position.
//!
//! Inputs: X, Scale, [Bias]
//! Output: Y (same shape as X)

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct LayerNorm {
    inputs: Vec<String>,
    axis: i64,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(inputs: Vec<String>, axis: i64, epsilon: f32) -> Self {
        Self { inputs, axis, epsilon }
    }
}

impl Layer for LayerNorm {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let x = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let bias = self.inputs.get(2).filter(|s| !s.is_empty())
            .and_then(|name| values.get(name));

        let rank = x.dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Number of elements in the normalised slice.
        let norm_size: usize = x.dims[axis..].iter().product();
        // Number of independent normalisation groups.
        let outer: usize = x.dims[..axis].iter().product();

        let x_data = x.floats()?;
        let scale_data = scale.floats()?;
        let bias_data = bias.and_then(|b| b.floats().ok());
        let eps = self.epsilon;

        let numel = outer * norm_size;
        let out = output.as_mut_f32(numel);

        for o in 0..outer {
            let slice = &x_data[o * norm_size..(o + 1) * norm_size];

            // Mean.
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

            // Variance.
            let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                / norm_size as f32;

            let inv_std = (var + eps).sqrt().recip();

            for i in 0..norm_size {
                let norm = (slice[i] - mean) * inv_std;
                // scale_data is over the normalised dims; if norm_size ==
                // scale_data.len() we index directly, otherwise broadcast.
                let s = scale_data[i % scale_data.len()];
                let b = bias_data.map(|bd| bd[i % bd.len()]).unwrap_or(0.0);
                out[o * norm_size + i] = norm * s + b;
            }
        }

        output.set_dims(&x.dims);
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

    fn run(x: Tensor, scale: Tensor, bias: Option<Tensor>, axis: i64) -> Vec<f32> {
        let mut intermediates = HashMap::new();
        intermediates.insert("x".into(), x);
        intermediates.insert("scale".into(), scale);
        let mut inputs = vec!["x".into(), "scale".into()];
        if let Some(b) = bias {
            intermediates.insert("bias".into(), b);
            inputs.push("bias".into());
        }
        let values = Values { intermediates: &intermediates, constants: Constants::empty() };
        let mut out = Tensor::default();
        let mut op = LayerNorm::new(inputs, axis, 1e-5);
        op.execute(&values, &mut out).unwrap();
        out.floats().unwrap().to_vec()
    }

    #[test]
    fn normalises_last_dim() {
        // [1, 4] input: mean=2.5, var=1.25 → each element normalised, then scaled by 1.
        let x = Tensor::new(dims![1, 4], vec![1.0f32, 2.0, 3.0, 4.0]);
        let scale = Tensor::new(dims![4], vec![1.0f32; 4]);
        let got = run(x, scale, None, -1);
        let mean: f32 = got.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be 0 after norm, got {mean}");
        let std: f32 = (got.iter().map(|v| v.powi(2)).sum::<f32>() / 4.0).sqrt();
        assert!((std - 1.0).abs() < 1e-3, "std should be ≈1, got {std}");
    }

    #[test]
    fn scale_and_bias_applied() {
        // All-same input → after normalisation everything is 0.  Bias=2 gives 2.
        let x = Tensor::new(dims![1, 3], vec![5.0f32, 5.0, 5.0]);
        let scale = Tensor::new(dims![3], vec![3.0f32; 3]);
        let bias = Tensor::new(dims![3], vec![2.0f32; 3]);
        let got = run(x, scale, Some(bias), -1);
        for &v in &got {
            assert!((v - 2.0).abs() < 1e-5, "expected 2, got {v}");
        }
    }

    #[test]
    fn axis_minus_one_matches_axis_rank_minus_one() {
        let x = Tensor::new(dims![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scale = Tensor::new(dims![3], vec![1.0f32; 3]);
        let a = run(x.clone(), scale.clone(), None, -1);
        let b = run(x, scale, None, 1);
        for (a, b) in a.iter().zip(b.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
