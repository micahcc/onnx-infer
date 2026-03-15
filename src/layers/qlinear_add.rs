use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::broadcast_index;
use crate::broadcast_shape;
use crate::get_tensor;
use crate::layers::Layer;

pub struct QLinearAdd {
    pub inputs: Vec<String>,
}

impl QLinearAdd {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for QLinearAdd {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_quant = get_tensor(values, &self.inputs[0])?;
        let x_scale = get_tensor(values, &self.inputs[1])?.floats()[0];
        let x_zp = get_tensor(values, &self.inputs[2])?.floats()[0];
        let y_quant = get_tensor(values, &self.inputs[3])?;
        let y_scale = get_tensor(values, &self.inputs[4])?.floats()[0];
        let y_zp = get_tensor(values, &self.inputs[5])?.floats()[0];
        let z_scale = get_tensor(values, &self.inputs[6])?.floats()[0];
        let z_zp = get_tensor(values, &self.inputs[7])?.floats()[0];

        let x_float_data = crate::layers::dequantize(x_quant.floats(), x_scale, x_zp);
        let y_float_data = crate::layers::dequantize(y_quant.floats(), y_scale, y_zp);

        let out_shape = broadcast_shape(&x_quant.dims, &y_quant.dims);
        let numel: usize = out_shape.iter().product();
        let ndim = out_shape.len();
        let mut index = vec![0usize; ndim];
        let mut z_float = vec![0.0f32; numel];

        for val in &mut z_float {
            let ai = broadcast_index(&index, &x_quant.dims, &out_shape);
            let bi = broadcast_index(&index, &y_quant.dims, &out_shape);
            *val = x_float_data[ai] + y_float_data[bi];
            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < out_shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }

        let z_quant = crate::layers::quantize_u8(&z_float, z_scale, z_zp);
        output.dims = out_shape;
        output.data_replace_f32(z_quant);
        Ok(())
    }
}
