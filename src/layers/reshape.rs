use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Reshape {
    pub inputs: Vec<String>,
    pub shape_attr: Option<Vec<i64>>,
}

impl Reshape {
    pub fn new(inputs: Vec<String>, shape_attr: Option<Vec<i64>>) -> Self {
        Self { inputs, shape_attr }
    }
}

impl Layer for Reshape {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;

        let shape_from_attr;
        let new_shape: &[i64] = if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let shape_tensor = get_tensor(values, &self.inputs[1])?;
            shape_tensor.ints().context("in Reshape layer")?
        } else {
            shape_from_attr = self
                .shape_attr
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Reshape: no shape input or attribute"))?;
            shape_from_attr
        };

        let total = input.numel();
        let mut dims = [0usize; 8];
        let dim_count = new_shape.len();
        let mut infer_idx: Option<usize> = None;

        for (i, &s) in new_shape.iter().enumerate() {
            if s == -1 {
                infer_idx = Some(i);
                dims[i] = 0;
            } else if s == 0 {
                dims[i] = input.dims[i];
            } else {
                dims[i] = s as usize;
            }
        }

        if let Some(idx) = infer_idx {
            let known: usize = dims[..dim_count]
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &v)| v)
                .product();
            if let Some(v) = total.checked_div(known) {
                dims[idx] = v;
            }
        }

        // If the shape was a static initializer with no -1 or 0, the total may
        // not match the actual input (e.g., batch dimension changed). Re-infer dim 0.
        if infer_idx.is_none() && !new_shape.contains(&0) {
            let shape_total: usize = dims[..dim_count].iter().product();
            if shape_total != total && shape_total > 0 && dim_count > 1 {
                let known: usize = dims[1..dim_count].iter().product();
                if known > 0 && total % known == 0 {
                    dims[0] = total / known;
                }
            }
        }
        output.copy_from(input);
        output.set_dims(&dims[..dim_count]);
        Ok(())
    }
}
