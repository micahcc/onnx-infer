use std::collections::HashMap;

use crate::Constants;
use crate::Result;
use crate::Tensor;

/// A read-only view over all tensor maps: intermediates + constants.
/// Layers look up tensors through this without needing to know which map holds them.
pub struct Values<'a> {
    pub intermediates: &'a HashMap<String, Tensor>,
    pub constants: Constants<'a>,
}

impl<'a> Values<'a> {
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.intermediates
            .get(name)
            .or_else(|| self.constants.get(name))
    }
}

pub fn get_tensor<'v>(values: &'v Values, name: &str) -> Result<&'v Tensor> {
    values
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' not found"))
}
