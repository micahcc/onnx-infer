use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Resize {
    pub inputs: Vec<String>,
}

impl Resize {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Resize {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();

        let mut out_dims = [0usize; 8];
        if self.inputs.len() > 3 && !self.inputs[3].is_empty() {
            let sizes = get_tensor(values, &self.inputs[3])?;
            for (i, &v) in sizes.ints().iter().enumerate() {
                out_dims[i] = v as usize;
            }
        } else if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            let scales = get_tensor(values, &self.inputs[2])?;
            let scales_f = scales.floats();
            for (i, (&d, &s)) in input.dims.iter().zip(scales_f.iter()).enumerate() {
                out_dims[i] = (d as f32 * s) as usize;
            }
        } else {
            return Err(InferenceError::InvalidModel(
                "Resize: no scales or sizes".into(),
            ));
        }

        let numel: usize = out_dims[..rank].iter().product();

        let mut out_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }
        let mut in_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        let input_f = input.floats();
        let buf = output.as_mut_f32(numel);
        #[allow(clippy::needless_range_loop)]
        for out_flat in 0..numel {
            let mut remaining = out_flat;
            let mut in_flat = 0;
            for ax in 0..rank {
                let out_coord = remaining / out_strides[ax];
                remaining %= out_strides[ax];
                let scale = out_dims[ax] as f32 / input.dims[ax] as f32;
                let in_coord = ((out_coord as f32 + 0.5) / scale - 0.5)
                    .round()
                    .max(0.0)
                    .min((input.dims[ax] - 1) as f32) as usize;
                in_flat += in_coord * in_strides[ax];
            }
            buf[out_flat] = input_f[in_flat];
        }

        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
