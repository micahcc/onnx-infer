use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct DequantizeLinear {
    pub inputs: Vec<String>,
    pub axis: i64,
    input_buf: Tensor,
    zp_buf: Tensor,
}

impl DequantizeLinear {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self {
            inputs,
            axis,
            input_buf: Tensor::default(),
            zp_buf: Tensor::default(),
        }
    }
}

impl Layer for DequantizeLinear {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let src_input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let src_zp = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            get_tensor(values, &self.inputs[2])?
        } else {
            &Tensor::new(vec![], vec![0.0])
        };

        // Pre-cast input and zero_point to f32 for fast inner loop
        self.input_buf.copy_cast_f32(src_input);
        self.zp_buf.copy_cast_f32(src_zp);

        let rank = self.input_buf.dims.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        let numel = self.input_buf.numel();
        let scale_f = scale.floats();
        let input_f = self.input_buf.floats();
        let zp_f = self.zp_buf.floats();

        if scale_f.len() == 1 {
            let s = scale_f[0];
            let zp = zp_f[0];
            let buf = output.as_mut_f32(numel);
            for i in 0..numel {
                buf[i] = (input_f[i] - zp) * s;
            }
            output.dims.clone_from(&self.input_buf.dims);
        } else {
            let outer: usize = self.input_buf.dims[..axis].iter().product();
            let ch = self.input_buf.dims[axis];
            let inner: usize = self.input_buf.dims[axis + 1..].iter().product();
            let buf = output.as_mut_f32(numel);
            for o in 0..outer {
                for c in 0..ch {
                    let s = scale_f[c];
                    let zp = zp_f[c];
                    let base = (o * ch + c) * inner;
                    for i in 0..inner {
                        buf[base + i] = (input_f[base + i] - zp) * s;
                    }
                }
            }
            output.dims.clone_from(&self.input_buf.dims);
        }
        Ok(())
    }
}
