use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct DequantizePrecomp {
    pub axis: usize,
    pub outer: usize,
    pub ch: usize,
    pub inner: usize,
}

pub struct DequantizeLinear {
    pub inputs: Vec<String>,
    pub axis: i64,
    default_zp: Tensor,
    pub precomp: Option<DequantizePrecomp>,
}

impl DequantizeLinear {
    pub fn new(inputs: Vec<String>, axis: i64, input_shape: &[usize]) -> Self {
        let precomp = if !input_shape.is_empty() {
            let shape = input_shape;
            let rank = shape.len() as i64;
            let a = if axis < 0 {
                (rank + axis) as usize
            } else {
                axis as usize
            };
            Some(DequantizePrecomp {
                axis: a,
                outer: shape[..a].iter().product(),
                ch: shape[a],
                inner: shape[a + 1..].iter().product(),
            })
        } else {
            None
        };

        Self {
            inputs,
            axis,
            default_zp: Tensor::new(crate::dims![], vec![0.0]),
            precomp,
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
            output.set_dims(&input.dims);
        } else {
            let (outer, ch, inner) = if let Some(p) = &self.precomp {
                (p.outer, p.ch, p.inner)
            } else {
                let rank = input.dims.len() as i64;
                let axis = if self.axis < 0 {
                    (rank + self.axis) as usize
                } else {
                    self.axis as usize
                };
                (
                    input.dims[..axis].iter().product(),
                    input.dims[axis],
                    input.dims[axis + 1..].iter().product(),
                )
            };
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
            output.set_dims(&input.dims);
        }
        Ok(())
    }
}
