use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::layers::Layer;
use crate::onnx::NodeProto;

pub struct Squeeze {
    pub inputs: Vec<String>,
    pub axes_attr: Vec<i64>,
}

impl Squeeze {
    pub fn new(inputs: Vec<String>, node: &NodeProto) -> Self {
        Self {
            inputs,
            axes_attr: get_attr_ints(node, "axes").unwrap_or_default(),
        }
    }
}

impl Layer for Squeeze {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let axes: &[i64] = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_tensor = get_tensor(values, &self.inputs[1])?;
            axes_tensor.ints()
        } else {
            &self.axes_attr
        };

        let rank = input.dims.len() as i64;
        let axes_set: std::collections::HashSet<usize> = if axes.is_empty() {
            input
                .dims
                .iter()
                .enumerate()
                .filter(|(_, d)| **d == 1)
                .map(|(i, _)| i)
                .collect()
        } else {
            axes.iter()
                .map(|&a| {
                    if a < 0 {
                        (rank + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        };

        let out_dims: Vec<usize> = input
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes_set.contains(i))
            .map(|(_, &d)| d)
            .collect();

        output.copy_from(input);
        output.dims = out_dims;
        Ok(())
    }
}
