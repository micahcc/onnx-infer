use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::conv::Conv;

pub struct QLinearConv {
    pub inputs: Vec<String>,
    pub inner: Conv,
    tmp_values: HashMap<String, Tensor>,
    conv_output: Tensor,
}

impl QLinearConv {
    pub fn new(inputs: Vec<String>, inner: Conv) -> Self {
        let mut tmp_values = HashMap::new();
        tmp_values.insert(inner.inputs[0].clone(), Tensor::default());
        tmp_values.insert(inner.inputs[1].clone(), Tensor::default());
        if inner.inputs.len() > 2 {
            tmp_values.insert(inner.inputs[2].clone(), Tensor::default());
        }
        Self {
            inputs,
            inner,
            tmp_values,
            conv_output: Tensor::default(),
        }
    }
}

impl Layer for QLinearConv {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_quant = get_tensor(values, &self.inputs[0])?;
        let x_scale = get_tensor(values, &self.inputs[1])?.floats()[0];
        let x_zp = get_tensor(values, &self.inputs[2])?.floats()[0];
        let w_quant = get_tensor(values, &self.inputs[3])?;
        let w_scale_t = get_tensor(values, &self.inputs[4])?;
        let w_zp_t = get_tensor(values, &self.inputs[5])?;
        let y_scale = get_tensor(values, &self.inputs[6])?.floats()[0];
        let y_zp = get_tensor(values, &self.inputs[7])?.floats()[0];
        let bias = if self.inputs.len() > 8 && !self.inputs[8].is_empty() {
            Some(get_tensor(values, &self.inputs[8])?)
        } else {
            None
        };

        // Dequantize X into scratch
        let x_tensor = self.tmp_values.get_mut(&self.inner.inputs[0]).unwrap();
        x_tensor.set_dims(&x_quant.dims);
        let x_buf = x_tensor.as_mut_f32(x_quant.numel());
        crate::layers::dequantize_into(x_quant.floats(), x_scale, x_zp, x_buf);

        // Dequantize W into scratch (per-channel)
        let c_out = w_quant.dims[0];
        let w_scale_f = w_scale_t.floats();
        let per_channel = w_scale_f.len() > 1;
        let elems_per_oc = w_quant.numel() / c_out;
        let w_quant_f = w_quant.floats();
        let w_tensor = self.tmp_values.get_mut(&self.inner.inputs[1]).unwrap();
        w_tensor.set_dims(&w_quant.dims);
        let w_buf = w_tensor.as_mut_f32(w_quant.numel());
        for oc in 0..c_out {
            let scale = if per_channel { w_scale_f[oc] } else { w_scale_f[0] };
            let zp = if per_channel { w_zp_t.f32_at(oc) } else { w_zp_t.f32_at(0) };
            let base = oc * elems_per_oc;
            for i in 0..elems_per_oc {
                w_buf[base + i] = (w_quant_f[base + i] - zp) * scale;
            }
        }

        // Dequantize bias into scratch if present
        if let Some(b) = bias {
            let b_tensor = self.tmp_values.get_mut(&self.inner.inputs[2]).unwrap();
            b_tensor.set_dims(&b.dims);
            let b_buf = b_tensor.as_mut_f32(b.numel());
            for oc in 0..b.numel() {
                let val = b.f32_at(oc);
                let ws = if per_channel { w_scale_f[oc] } else { w_scale_f[0] };
                b_buf[oc] = val * x_scale * ws;
            }
        }

        self.inner.execute(&self.tmp_values, &mut self.conv_output)?;

        let numel = self.conv_output.numel();
        output.set_dims(&self.conv_output.dims);
        let out_buf = output.as_mut_f32(numel);
        crate::layers::quantize_u8_into(self.conv_output.floats(), y_scale, y_zp, out_buf);
        Ok(())
    }
}
