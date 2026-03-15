use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;
use crate::layers::global_avg_pool::GlobalAvgPool;

pub struct QLinearGlobalAvgPool {
    pub inputs: Vec<String>,
    pub inner: GlobalAvgPool,
}

impl QLinearGlobalAvgPool {
    pub fn new(inputs: Vec<String>, inner: GlobalAvgPool) -> Self {
        Self { inputs, inner }
    }
}

impl Layer for QLinearGlobalAvgPool {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_quant = get_tensor(values, &self.inputs[0])?;
        let x_scale = get_tensor(values, &self.inputs[1])?.floats()[0];
        let x_zp = get_tensor(values, &self.inputs[2])?.floats()[0];
        let y_scale = get_tensor(values, &self.inputs[3])?.floats()[0];
        let y_zp = get_tensor(values, &self.inputs[4])?.floats()[0];

        let x_float = Tensor::new(
            x_quant.dims.clone(),
            crate::layers::dequantize(x_quant.floats(), x_scale, x_zp),
        );

        let mut tmp_values: HashMap<String, Tensor> = HashMap::new();
        tmp_values.insert(self.inner.inputs[0].clone(), x_float);

        let mut gap_output = Tensor::default();
        self.inner.execute(&tmp_values, &mut gap_output)?;

        let y_quant = crate::layers::quantize_u8(gap_output.floats(), y_scale, y_zp);
        output.dims = gap_output.dims;
        output.data_replace_f32(y_quant);
        Ok(())
    }
}
