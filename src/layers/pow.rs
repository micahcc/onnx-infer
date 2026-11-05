use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Pow {
    pub inputs: Vec<String>,
    pub legacy_broadcast: bool,
    pub axis: usize,
}

impl Pow {
    pub fn new(inputs: Vec<String>, legacy_broadcast: bool, axis: usize) -> Self {
        Self { inputs, legacy_broadcast, axis }
    }
}

impl Layer for Pow {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;
        crate::layers::binary_op(a, b, output, self.legacy_broadcast, self.axis, |a, b| a.powf(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constants;
    use crate::Tensor;
    use crate::Values;
    use crate::dims;
    use std::collections::HashMap;

    fn make_values(pairs: Vec<(&str, Tensor)>) -> HashMap<String, Tensor> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn run_pow(a: Tensor, b: Tensor) -> Vec<f32> {
        let intermediates = make_values(vec![("a", a), ("b", b)]);
        let vals = Values { intermediates: &intermediates, constants: Constants::empty() };
        let mut layer = Pow::new(vec!["a".to_string(), "b".to_string()], false, 0);
        let mut out = Tensor::new(dims![1], vec![0.0f32]);
        layer.execute(&vals, &mut out).unwrap();
        out.floats().unwrap().to_vec()
    }

    #[test]
    fn integer_exponent() {
        let a = Tensor::new(dims![3], vec![2.0f32, 3.0, 4.0]);
        let b = Tensor::new(dims![3], vec![3.0f32, 2.0, 0.5]);
        let result = run_pow(a, b);
        assert!((result[0] - 8.0).abs() < 1e-5, "2^3 = {}", result[0]);
        assert!((result[1] - 9.0).abs() < 1e-5, "3^2 = {}", result[1]);
        assert!((result[2] - 2.0).abs() < 1e-5, "4^0.5 = {}", result[2]);
    }

    #[test]
    fn scalar_exponent_broadcast() {
        let a = Tensor::new(dims![3], vec![2.0f32, 3.0, 4.0]);
        let b = Tensor::new(dims![1], vec![2.0f32]);
        let result = run_pow(a, b);
        assert!((result[0] - 4.0).abs() < 1e-5);
        assert!((result[1] - 9.0).abs() < 1e-5);
        assert!((result[2] - 16.0).abs() < 1e-5);
    }
}
