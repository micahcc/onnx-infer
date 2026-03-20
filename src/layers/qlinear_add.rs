use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape_into;
use crate::get_tensor;
use crate::layers::Layer;

pub struct QLinearAdd {
    pub inputs: Vec<String>,
    x_scratch: Vec<f32>,
    y_scratch: Vec<f32>,
}

impl QLinearAdd {
    pub fn new(inputs: Vec<String>) -> Self {
        Self {
            inputs,
            x_scratch: Vec::new(),
            y_scratch: Vec::new(),
        }
    }
}

impl Layer for QLinearAdd {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_quant = get_tensor(values, &self.inputs[0])?;
        let x_scale = get_tensor(values, &self.inputs[1])?.floats().context("in QLinearAdd layer")?[0];
        let x_zp = get_tensor(values, &self.inputs[2])?.floats().context("in QLinearAdd layer")?[0];
        let y_quant = get_tensor(values, &self.inputs[3])?;
        let y_scale = get_tensor(values, &self.inputs[4])?.floats().context("in QLinearAdd layer")?[0];
        let y_zp = get_tensor(values, &self.inputs[5])?.floats().context("in QLinearAdd layer")?[0];
        let z_scale = get_tensor(values, &self.inputs[6])?.floats().context("in QLinearAdd layer")?[0];
        let z_zp = get_tensor(values, &self.inputs[7])?.floats().context("in QLinearAdd layer")?[0];

        let x_numel = x_quant.numel();
        let y_numel = y_quant.numel();
        self.x_scratch.resize(x_numel, 0.0);
        self.y_scratch.resize(y_numel, 0.0);
        crate::layers::dequantize_into(x_quant.floats().context("in QLinearAdd layer")?, x_scale, x_zp, &mut self.x_scratch);
        crate::layers::dequantize_into(y_quant.floats().context("in QLinearAdd layer")?, y_scale, y_zp, &mut self.y_scratch);

        let ndim = x_quant.dims.len().max(y_quant.dims.len());
        let mut out_shape = [0usize; 8];
        broadcast_shape_into(&x_quant.dims, &y_quant.dims, &mut out_shape[..ndim]);
        let numel: usize = out_shape[..ndim].iter().product();
        let mut index = [0usize; 8];

        // Add into output buffer directly, then quantize in-place
        let buf = output.as_mut_f32(numel);
        for val in buf.iter_mut() {
            let ai = broadcast_index(&index[..ndim], &x_quant.dims, &out_shape[..ndim]);
            let bi = broadcast_index(&index[..ndim], &y_quant.dims, &out_shape[..ndim]);
            *val = self.x_scratch[ai] + self.y_scratch[bi];
            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < out_shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }

        // Quantize in-place
        for val in buf.iter_mut() {
            *val = (*val / z_scale + z_zp).round().clamp(0.0, 255.0);
        }

        output.set_dims(&out_shape[..ndim]);
        Ok(())
    }
}
