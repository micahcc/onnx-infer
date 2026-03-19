use std::collections::HashMap;

use crate::Dims;
use crate::InferenceError;
use crate::Result;
use crate::Tensor;

pub fn get_tensor<'a>(values: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    values
        .get(name)
        .ok_or_else(|| InferenceError::InvalidModel(format!("Tensor '{name}' not found")))
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
