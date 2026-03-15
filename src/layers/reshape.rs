use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::layers::Layer;
use crate::onnx::NodeProto;

pub struct Reshape {
    pub inputs: Vec<String>,
    pub shape_attr: Option<Vec<i64>>,
}

impl Reshape {
    pub fn new(inputs: Vec<String>, node: &NodeProto) -> Self {
        Self {
            inputs,
            shape_attr: get_attr_ints(node, "shape"),
        }
    }
}

impl Layer for Reshape {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let shape_from_attr;
        let new_shape: &[i64] = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let shape_tensor = get_tensor(values, &self.inputs[1])?;
            shape_tensor.ints()
        } else {
            shape_from_attr = self.shape_attr.as_ref().ok_or_else(|| {
                InferenceError::InvalidModel("Reshape: no shape input or attribute".into())
            })?;
            shape_from_attr
        };

        let total = input.numel();
        let mut dims: Vec<usize> = Vec::new();
        let mut infer_idx: Option<usize> = None;

        for (i, &s) in new_shape.iter().enumerate() {
            if s == -1 {
                infer_idx = Some(i);
                dims.push(0);
            } else if s == 0 {
                dims.push(input.dims[i]);
            } else {
                dims.push(s as usize);
            }
        }

        if let Some(idx) = infer_idx {
            let known: usize = dims
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &v)| v)
                .product();
            dims[idx] = total / known;
        }

        output.copy_from(input);
        output.dims = dims;
        Ok(())
    }
}
