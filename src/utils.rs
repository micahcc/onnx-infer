use std::collections::HashMap;

use crate::Dims;
use crate::Result;
use crate::Tensor;

/// A read-only view over multiple tensor maps: intermediates, inputs, and initializers.
/// Layers look up tensors through this without needing to know which map holds them.
pub struct Values<'a> {
    pub intermediates: &'a HashMap<String, Tensor>,
    pub inputs: &'a HashMap<String, Tensor>,
    pub initializers: &'a HashMap<String, Tensor>,
}

impl<'a> Values<'a> {
    pub fn get(&self, name: &str) -> Option<&'a Tensor> {
        self.intermediates
            .get(name)
            .or_else(|| self.inputs.get(name))
            .or_else(|| self.initializers.get(name))
    }
}

pub fn get_tensor<'a>(values: &Values<'a>, name: &str) -> Result<&'a Tensor> {
    values
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' not found"))
}

pub fn get_tensor_map<'a>(
    values: &'a HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor> {
    values
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' not found"))
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Dims {
    let max_len = a.len().max(b.len());
    let mut result = crate::dims![1usize; max_len];
    broadcast_shape_into(a, b, &mut result);
    result
}

pub fn broadcast_shape_into(a: &[usize], b: &[usize], out: &mut [usize]) {
    let max_len = out.len();
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        out[i] = if da == 0 || db == 0 { 0 } else { da.max(db) };
    }
}

pub fn broadcast_index(index: &[usize], shape: &[usize], out_shape: &[usize]) -> usize {
    let offset = out_shape.len() - shape.len();
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        let dim_idx = if shape[i] == 1 { 0 } else { index[i + offset] };
        flat += dim_idx * stride;
        stride *= shape[i];
    }
    flat
}
