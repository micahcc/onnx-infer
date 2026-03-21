use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

/// CategoryMapper (ai.onnx.ml): maps string inputs to int64 using a lookup table.
pub struct CategoryMapper {
    pub inputs: Vec<String>,
    pub map: HashMap<Vec<u8>, i64>,
    pub default_int64: i64,
}

impl CategoryMapper {
    pub fn new(
        inputs: Vec<String>,
        cats_strings: Vec<Vec<u8>>,
        cats_int64s: Vec<i64>,
        default_int64: i64,
    ) -> Self {
        let mut map = HashMap::with_capacity(cats_strings.len());
        for (s, &i) in cats_strings.iter().zip(cats_int64s.iter()) {
            map.insert(s.clone(), i);
        }
        Self {
            inputs,
            map,
            default_int64,
        }
    }
}

impl Layer for CategoryMapper {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let strings = input.strings().context("in CategoryMapper layer")?;
        let numel = strings.len();
        let buf = output.as_mut_i64(numel);
        for (o, s) in buf.iter_mut().zip(strings.iter()) {
            *o = self.map.get(s).copied().unwrap_or(self.default_int64);
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
