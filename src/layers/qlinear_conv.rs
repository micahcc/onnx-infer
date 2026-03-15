use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::conv::Conv;

pub struct QLinearConv {
    pub inputs: Vec<String>,
    pub inner: Conv,
}

impl QLinearConv {
    pub fn new(inputs: Vec<String>, inner: Conv) -> Self {
        Self { inputs, inner }
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

        let x_float = Tensor::new(
            x_quant.dims.clone(),
            crate::layers::dequantize(x_quant.floats(), x_scale, x_zp),
        );

        let c_out = w_quant.dims[0];
        let w_scale_f = w_scale_t.floats();
        let per_channel = w_scale_f.len() > 1;
        let elems_per_oc = w_quant.numel() / c_out;
        let w_quant_f = w_quant.floats();
        let mut w_float_data = vec![0.0f32; w_quant.numel()];
        for oc in 0..c_out {
            let scale = if per_channel {
                w_scale_f[oc]
            } else {
                w_scale_f[0]
            };
            let zp = if per_channel {
                w_zp_t.f32_at(oc)
            } else {
                w_zp_t.f32_at(0)
            };
            let base = oc * elems_per_oc;
            for i in 0..elems_per_oc {
                w_float_data[base + i] = (w_quant_f[base + i] - zp) * scale;
            }
        }
        let w_float = Tensor::new(w_quant.dims.clone(), w_float_data);

        let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
        tmp_values.insert(self.inner.inputs[0].clone(), x_float);
        tmp_values.insert(self.inner.inputs[1].clone(), w_float);
        if let Some(b) = bias {
            let bias_float: Vec<f32> = (0..b.numel())
                .map(|oc| {
                    let val = b.f32_at(oc);
                    let ws = if per_channel {
                        w_scale_f[oc]
                    } else {
                        w_scale_f[0]
                    };
                    val * x_scale * ws
                })
                .collect();
            tmp_values.insert(
                self.inner.inputs[2].clone(),
                Tensor::new(b.dims.clone(), bias_float),
            );
        }

        let mut conv_output = Tensor::default();
        self.inner.execute(&tmp_values, &mut conv_output)?;

        let y_quant = crate::layers::quantize_u8(conv_output.floats(), y_scale, y_zp);
        output.dims = conv_output.dims;
        output.data_replace_f32(y_quant);
        Ok(())
    }
}
