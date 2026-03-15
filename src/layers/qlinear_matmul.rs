use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::matmul::MatMul;

pub struct QLinearMatMul {
    pub inputs: Vec<String>,
    pub inner: MatMul,
    tmp_values: HashMap<String, Tensor>,
    mm_output: Tensor,
}

impl QLinearMatMul {
    pub fn new(inputs: Vec<String>, inner: MatMul) -> Self {
        let mut tmp_values = HashMap::new();
        tmp_values.insert(inner.inputs[0].clone(), Tensor::default());
        tmp_values.insert(inner.inputs[1].clone(), Tensor::default());
        Self {
            inputs,
            inner,
            tmp_values,
            mm_output: Tensor::default(),
        }
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

        // Dequantize A into scratch
        let a_tensor = self.tmp_values.get_mut(&self.inner.inputs[0]).unwrap();
        a_tensor.set_dims(&a_quant.dims);
        let a_buf = a_tensor.as_mut_f32(a_quant.numel());
        crate::layers::dequantize_into(a_quant.floats(), a_scale, a_zp, a_buf);

        // Dequantize B into scratch
        let b_scale_f = b_scale_t.floats();
        let b_quant_f = b_quant.floats();
        let b_tensor = self.tmp_values.get_mut(&self.inner.inputs[1]).unwrap();
        b_tensor.set_dims(&b_quant.dims);
        let b_buf = b_tensor.as_mut_f32(b_quant.numel());
        if b_scale_f.len() > 1 {
            let n = *b_quant.dims.last().unwrap();
            let k = b_quant.numel() / n;
            for row in 0..k {
                for col in 0..n {
                    let idx = row * n + col;
                    b_buf[idx] = (b_quant_f[idx] - b_zp_t.f32_at(col)) * b_scale_f[col];
                }
            }
        } else {
            crate::layers::dequantize_into(b_quant_f, b_scale_f[0], b_zp_t.f32_at(0), b_buf);
        }

        self.inner.execute(&self.tmp_values, &mut self.mm_output)?;

        let numel = self.mm_output.numel();
        output.set_dims(&self.mm_output.dims);
        let out_buf = output.as_mut_f32(numel);
        crate::layers::quantize_u8_into(self.mm_output.floats(), y_scale, y_zp, out_buf);
        Ok(())
    }
}
