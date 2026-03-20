use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::global_avg_pool::GlobalAvgPool;

pub struct QLinearGlobalAvgPool {
    pub inputs: Vec<String>,
    pub inner: GlobalAvgPool,
    tmp_values: HashMap<String, Tensor>,
    gap_output: Tensor,
}

impl QLinearGlobalAvgPool {
    pub fn new(inputs: Vec<String>, inner: GlobalAvgPool) -> Self {
        let mut tmp_values = HashMap::new();
        tmp_values.insert(inner.inputs[0].clone(), Tensor::default());
        Self {
            inputs,
            inner,
            tmp_values,
            gap_output: Tensor::default(),
        }
    }
}

impl Layer for QLinearGlobalAvgPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_quant = get_tensor(values, &self.inputs[0])?;
        let x_scale = get_tensor(values, &self.inputs[1])?.floats().context("in QLinearGlobalAvgPool layer")?[0];
        let x_zp = get_tensor(values, &self.inputs[2])?.floats().context("in QLinearGlobalAvgPool layer")?[0];
        let y_scale = get_tensor(values, &self.inputs[3])?.floats().context("in QLinearGlobalAvgPool layer")?[0];
        let y_zp = get_tensor(values, &self.inputs[4])?.floats().context("in QLinearGlobalAvgPool layer")?[0];

        let x_tensor = self.tmp_values.get_mut(&self.inner.inputs[0]).unwrap();
        x_tensor.set_dims(&x_quant.dims);
        let x_buf = x_tensor.as_mut_f32(x_quant.numel());
        crate::layers::dequantize_into(x_quant.floats().context("in QLinearGlobalAvgPool layer")?, x_scale, x_zp, x_buf);

        self.inner.execute(&self.tmp_values, &mut self.gap_output)?;

        let numel = self.gap_output.numel();
        output.set_dims(&self.gap_output.dims);
        let out_buf = output.as_mut_f32(numel);
        crate::layers::quantize_u8_into(self.gap_output.floats().context("in QLinearGlobalAvgPool layer")?, y_scale, y_zp, out_buf);
        Ok(())
    }
}
