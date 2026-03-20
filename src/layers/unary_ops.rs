use anyhow::Context;
use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

macro_rules! unary_op {
    ($name:ident, |$v:ident| $body:expr) => {
        pub struct $name {
            pub inputs: Vec<String>,
        }

        impl $name {
            pub fn new(inputs: Vec<String>) -> Self {
                Self { inputs }
            }
        }

        impl Layer for $name {
            fn execute(
                &mut self,
                values: &HashMap<String, Tensor>,
                output: &mut Tensor,
            ) -> Result<()> {
                let input = get_tensor(values, &self.inputs[0])?;
                let inp = input.floats().context("in UnaryOps layer")?;
                let buf = output.as_mut_f32(inp.len());
                for (o, &$v) in buf.iter_mut().zip(inp.iter()) {
                    *o = $body;
                }
                output.set_dims(&input.dims);
                Ok(())
            }
        }
    };
}

// Trig
unary_op!(Sin, |v| v.sin());
unary_op!(Cos, |v| v.cos());
unary_op!(Tan, |v| v.tan());
unary_op!(Asin, |v| v.asin());
unary_op!(Acos, |v| v.acos());
unary_op!(Atan, |v| v.atan());

// Hyperbolic
unary_op!(Sinh, |v| v.sinh());
unary_op!(Cosh, |v| v.cosh());
unary_op!(Asinh, |v| v.asinh());
unary_op!(Acosh, |v| v.acosh());
unary_op!(Atanh, |v| v.atanh());

// Math
unary_op!(Erf, |v| {
    // Abramowitz & Stegun approximation (max error ~1.5e-7)
    let a1: f32 = 0.254_829_6;
    let a2: f32 = -0.284_496_72;
    let a3: f32 = 1.421_413_8;
    let a4: f32 = -1.453_152_1;
    let a5: f32 = 1.061_405_4;
    let p: f32 = 0.3275911;
    let sign = v.signum();
    let x = v.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
});
unary_op!(Sign, |v| v.signum());
unary_op!(Neg, |v| -v);
unary_op!(Reciprocal, |v| 1.0 / v);
unary_op!(Softsign, |v| v / (1.0 + v.abs()));
unary_op!(IsNaN, |v| if v.is_nan() { 1.0 } else { 0.0 });
unary_op!(IsInf, |v| if v.is_infinite() { 1.0 } else { 0.0 });

// Activations with parameters

pub struct Elu {
    pub inputs: Vec<String>,
    pub alpha: f32,
}

impl Elu {
    pub fn new(inputs: Vec<String>, alpha: f32) -> Self {
        Self { inputs, alpha }
    }
}

impl Layer for Elu {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in UnaryOps layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = if v >= 0.0 {
                v
            } else {
                self.alpha * (v.exp() - 1.0)
            };
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}

pub struct Celu {
    pub inputs: Vec<String>,
    pub alpha: f32,
}

impl Celu {
    pub fn new(inputs: Vec<String>, alpha: f32) -> Self {
        Self { inputs, alpha }
    }
}

impl Layer for Celu {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in UnaryOps layer")?;
        let buf = output.as_mut_f32(inp.len());
        let a = self.alpha;
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = v.max(0.0) + (a * ((v / a).exp() - 1.0)).min(0.0);
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}

pub struct Selu {
    pub inputs: Vec<String>,
    pub alpha: f32,
    pub gamma: f32,
}

impl Selu {
    pub fn new(inputs: Vec<String>, alpha: f32, gamma: f32) -> Self {
        Self {
            inputs,
            alpha,
            gamma,
        }
    }
}

impl Layer for Selu {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in UnaryOps layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = self.gamma
                * if v > 0.0 {
                    v
                } else {
                    self.alpha * (v.exp() - 1.0)
                };
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}

pub struct HardSigmoid {
    pub inputs: Vec<String>,
    pub alpha: f32,
    pub beta: f32,
}

impl HardSigmoid {
    pub fn new(inputs: Vec<String>, alpha: f32, beta: f32) -> Self {
        Self {
            inputs,
            alpha,
            beta,
        }
    }
}

impl Layer for HardSigmoid {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in UnaryOps layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = (self.alpha * v + self.beta).clamp(0.0, 1.0);
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}

pub struct ThresholdedRelu {
    pub inputs: Vec<String>,
    pub alpha: f32,
}

impl ThresholdedRelu {
    pub fn new(inputs: Vec<String>, alpha: f32) -> Self {
        Self { inputs, alpha }
    }
}

impl Layer for ThresholdedRelu {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let inp = input.floats().context("in UnaryOps layer")?;
        let buf = output.as_mut_f32(inp.len());
        for (o, &v) in buf.iter_mut().zip(inp.iter()) {
            *o = if v > self.alpha { v } else { 0.0 };
        }
        output.set_dims(&input.dims);
        Ok(())
    }
}
