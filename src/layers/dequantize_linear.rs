use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct DequantizeLinear {
    pub inputs: Vec<String>,
    pub axis: i64,
    default_zp: Tensor,
}

impl DequantizeLinear {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self {
            inputs,
            axis,
            default_zp: Tensor::new(vec![], vec![0.0]),
        }
    }
}

impl Layer for DequantizeLinear {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let scale = get_tensor(values, &self.inputs[1])?;
        let zp = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            get_tensor(values, &self.inputs[2])?
        } else {
            &self.default_zp
        };

        let rank = input.dims.len() as i64;
        let axis = if self.axis < 0 {
            (rank + self.axis) as usize
        } else {
            self.axis as usize
        };

        let numel = input.numel();
        let scale_f = scale.floats();
        let input_f = input.floats();
        let zp_f = zp.floats();

        if scale_f.len() == 1 {
            let s = scale_f[0];
            let z = zp_f[0];
            let buf = output.as_mut_f32(numel);
            for i in 0..numel {
                buf[i] = (input_f[i] - z) * s;
            }
            output.dims.clone_from(&input.dims);
        } else {
            let outer: usize = input.dims[..axis].iter().product();
            let ch = input.dims[axis];
            let inner: usize = input.dims[axis + 1..].iter().product();
            let buf = output.as_mut_f32(numel);
            for o in 0..outer {
                for c in 0..ch {
                    let s = scale_f[c];
                    let z = zp_f[c];
                    let base = (o * ch + c) * inner;
                    for i in 0..inner {
                        buf[base + i] = (input_f[base + i] - z) * s;
                    }
                }
            }
            output.dims.clone_from(&input.dims);
        }
        Ok(())
    }
}
