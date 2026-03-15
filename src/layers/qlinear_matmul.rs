use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::matmul::MatMul;

pub struct QLinearMatMul {
    pub inputs: Vec<String>,
    pub inner: MatMul,
}

impl QLinearMatMul {
    pub fn new(inputs: Vec<String>, inner: MatMul) -> Self {
        Self { inputs, inner }
    }
}

impl Layer for QLinearMatMul {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let a_quant = get_tensor(values, &self.inputs[0])?;
        let a_scale = get_tensor(values, &self.inputs[1])?.floats()[0];
        let a_zp = get_tensor(values, &self.inputs[2])?.floats()[0];
        let b_quant = get_tensor(values, &self.inputs[3])?;
        let b_scale_t = get_tensor(values, &self.inputs[4])?;
        let b_zp_t = get_tensor(values, &self.inputs[5])?;
        let y_scale = get_tensor(values, &self.inputs[6])?.floats()[0];
        let y_zp = get_tensor(values, &self.inputs[7])?.floats()[0];

        let a_float = Tensor::new(
            a_quant.dims.clone(),
            crate::layers::dequantize(a_quant.floats(), a_scale, a_zp),
        );

        let b_scale_f = b_scale_t.floats();
        let b_quant_f = b_quant.floats();
        let b_float = if b_scale_f.len() > 1 {
            let n = *b_quant.dims.last().unwrap();
            let k = b_quant.numel() / n;
            let mut data = vec![0.0f32; b_quant.numel()];
            for row in 0..k {
                for col in 0..n {
                    let idx = row * n + col;
                    data[idx] = (b_quant_f[idx] - b_zp_t.f32_at(col)) * b_scale_f[col];
                }
            }
            Tensor::new(b_quant.dims.clone(), data)
        } else {
            Tensor::new(
                b_quant.dims.clone(),
                crate::layers::dequantize(b_quant_f, b_scale_f[0], b_zp_t.f32_at(0)),
            )
        };

        let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
        tmp_values.insert(self.inner.inputs[0].clone(), a_float);
        tmp_values.insert(self.inner.inputs[1].clone(), b_float);

        let mut mm_output = Tensor::default();
        self.inner.execute(&tmp_values, &mut mm_output)?;

        let y_quant = crate::layers::quantize_u8(mm_output.floats(), y_scale, y_zp);
        output.dims = mm_output.dims;
        output.data_replace_f32(y_quant);
        Ok(())
    }
}
