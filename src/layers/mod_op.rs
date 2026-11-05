use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Mod {
    pub inputs: Vec<String>,
    pub legacy_broadcast: bool,
    pub axis: usize,
    pub fmod: bool,
}

impl Mod {
    pub fn new(inputs: Vec<String>, legacy_broadcast: bool, axis: usize, fmod: bool) -> Self {
        Self { inputs, legacy_broadcast, axis, fmod }
    }
}

impl Layer for Mod {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;
        if self.fmod {
            // C-style fmod: result has same sign as dividend
            crate::layers::binary_op(a, b, output, self.legacy_broadcast, self.axis, |a, b| a % b)
        } else {
            // Python/NumPy-style: result has same sign as divisor
            crate::layers::binary_op(a, b, output, self.legacy_broadcast, self.axis, |a, b| {
                ((a % b) + b) % b
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constants;
    use crate::Values;
    use crate::dims;
    use crate::Tensor;
    use std::collections::HashMap;

    fn run(a: Vec<f32>, b: Vec<f32>, fmod: bool) -> Vec<f32> {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("a".into(), Tensor::new(dims![3], a));
        m.insert("b".into(), Tensor::new(dims![3], b));
        let vals = Values { intermediates: &m, constants: Constants::empty() };
        let mut out = Tensor::new(dims![3], vec![0.0f32; 3]);
        let mut op = Mod::new(vec!["a".into(), "b".into()], false, 0, fmod);
        op.execute(&vals, &mut out).unwrap();
        out.floats().unwrap().to_vec()
    }

    #[test]
    fn fmod_positive() {
        let r = run(vec![5.0, 7.0, 9.0], vec![3.0, 3.0, 4.0], true);
        assert!((r[0] - 2.0).abs() < 1e-5, "5%3={}", r[0]);
        assert!((r[1] - 1.0).abs() < 1e-5, "7%3={}", r[1]);
        assert!((r[2] - 1.0).abs() < 1e-5, "9%4={}", r[2]);
    }

    #[test]
    fn python_mod() {
        let r = run(vec![5.0, 7.0, 9.0], vec![3.0, 3.0, 4.0], false);
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-5);
        assert!((r[2] - 1.0).abs() < 1e-5);
    }
}
