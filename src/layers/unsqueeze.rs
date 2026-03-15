use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::layers::Layer;
use crate::onnx::NodeProto;

pub struct Unsqueeze {
    pub inputs: Vec<String>,
    pub axes_attr: Vec<i64>,
}

impl Unsqueeze {
    pub fn new(inputs: Vec<String>, node: &NodeProto) -> Self {
        Self {
            inputs,
            axes_attr: get_attr_ints(node, "axes").unwrap_or_default(),
        }
    }
}

impl Layer for Unsqueeze {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let axes: &[i64] = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let axes_tensor = get_tensor(values, &self.inputs[1])?;
            axes_tensor.ints()
        } else {
            &self.axes_attr
        };

        let out_rank = input.dims.len() + axes.len();
        let mut out_dims = input.dims.clone();
        let mut sorted_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (out_rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        sorted_axes.sort();
        for &ax in &sorted_axes {
            out_dims.insert(ax, 1);
        }

        output.copy_from(input);
        output.dims = out_dims;
        Ok(())
    }
}
