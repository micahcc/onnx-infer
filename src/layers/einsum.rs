//! Einsum operator — general Einstein summation.
//!
//! Supports two-input equations with any number of free indices and any number
//! of contraction (summed) indices.  Each input dimension is identified by a
//! single letter; letters absent from the output are summed over.
//!
//! The only equation used in practice here is `bmc,bchw->bmhw` (contract `c`),
//! but the implementation is general for two-tensor equations.

use anyhow::bail;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct Einsum {
    inputs: Vec<String>,
    equation: String,
}

impl Einsum {
    pub fn new(inputs: Vec<String>, equation: String) -> Self {
        Self { inputs, equation }
    }
}

/// Parse `lhs1,lhs2->rhs` into (lhs1_chars, lhs2_chars, rhs_chars).
fn parse_equation(eq: &str) -> anyhow::Result<(Vec<char>, Vec<char>, Vec<char>)> {
    let (lhs, rhs) = match eq.split_once("->") {
        Some(p) => p,
        None => bail!("Einsum: equation must contain '->'"),
    };
    let operands: Vec<&str> = lhs.split(',').collect();
    if operands.len() != 2 {
        bail!("Einsum: only two-input equations are supported, got {}", operands.len());
    }
    Ok((
        operands[0].chars().collect(),
        operands[1].chars().collect(),
        rhs.chars().collect(),
    ))
}

impl Layer for Einsum {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let a = get_tensor(values, &self.inputs[0])?;
        let b = get_tensor(values, &self.inputs[1])?;

        let (a_idx, b_idx, out_idx) = parse_equation(&self.equation)?;

        // Collect all unique dimension labels and their sizes.
        let mut dim_size: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for (&ch, &sz) in a_idx.iter().zip(a.dims.iter()) {
            dim_size.insert(ch, sz);
        }
        for (&ch, &sz) in b_idx.iter().zip(b.dims.iter()) {
            dim_size.insert(ch, sz);
        }

        // Output shape.
        let out_shape: Vec<usize> = out_idx.iter().map(|c| dim_size[c]).collect();
        let out_numel: usize = out_shape.iter().product();

        // Contraction indices = labels in both inputs but not in output.
        let contract_labels: Vec<char> = dim_size
            .keys()
            .copied()
            .filter(|c| !out_idx.contains(c))
            .collect();

        // Build strides for each operand.
        let strides = |labels: &[char], dims: &[usize]| -> Vec<usize> {
            let n = labels.len();
            let mut s = vec![1usize; n];
            for i in (0..n.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * dims[i + 1];
            }
            s
        };
        let a_strides = strides(&a_idx, &a.dims);
        let b_strides = strides(&b_idx, &b.dims);

        let a_data = a.floats()?;
        let b_data = b.floats()?;

        // Output strides.
        let o_strides = strides(&out_idx, &out_shape);

        let buf = output.as_mut_f32(out_numel);
        buf.fill(0.0);

        // Enumerate all output coords.
        let out_rank = out_idx.len();
        let mut out_coords = vec![0usize; out_rank];

        // Also enumerate contraction coords.
        let contract_sizes: Vec<usize> = contract_labels.iter().map(|c| dim_size[c]).collect();
        let total_contract: usize = contract_sizes.iter().product::<usize>().max(1);

        // Helper: flat index for a tensor given a coord mapping char→usize.
        let flat_idx = |labels: &[char], strides: &[usize], coord: &dyn Fn(char) -> usize| -> usize {
            labels.iter().zip(strides.iter()).map(|(&c, &s)| coord(c) * s).sum()
        };

        loop {
            // Build out coord map: char → size
            let out_map: std::collections::HashMap<char, usize> = out_idx
                .iter()
                .copied()
                .zip(out_coords.iter().copied())
                .collect();

            let out_flat: usize = out_coords.iter().zip(o_strides.iter()).map(|(&c, &s)| c * s).sum();

            // Iterate over contraction dimensions.
            let mut c_coords = vec![0usize; contract_labels.len()];
            for _ in 0..total_contract {
                let c_map: std::collections::HashMap<char, usize> = contract_labels
                    .iter()
                    .copied()
                    .zip(c_coords.iter().copied())
                    .collect();

                let coord = |c: char| -> usize {
                    *out_map.get(&c).or_else(|| c_map.get(&c)).unwrap_or(&0)
                };

                let ai = flat_idx(&a_idx, &a_strides, &coord);
                let bi = flat_idx(&b_idx, &b_strides, &coord);
                buf[out_flat] += a_data[ai] * b_data[bi];

                // Advance contraction coords.
                if !c_coords.is_empty() {
                    for d in (0..c_coords.len()).rev() {
                        c_coords[d] += 1;
                        if c_coords[d] < contract_sizes[d] { break; }
                        c_coords[d] = 0;
                    }
                }
            }

            // Advance output coords.
            if out_rank == 0 { break; }
            let mut advanced = false;
            for d in (0..out_rank).rev() {
                out_coords[d] += 1;
                if out_coords[d] < out_shape[d] { advanced = true; break; }
                out_coords[d] = 0;
            }
            if !advanced { break; }
        }

        output.set_dims(&out_shape);
        Ok(())
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

    #[test]
    fn matrix_multiply_ij_jk_ik() {
        // ij,jk->ik  is matmul A(2,3) @ B(3,2) = C(2,2)
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("a".into(), Tensor::new(dims![2, 3], a));
        m.insert("b".into(), Tensor::new(dims![3, 2], b));
        let vals = Values { intermediates: &m, constants: Constants::empty() };
        let mut out = Tensor::new(dims![2, 2], vec![0.0f32; 4]);
        let mut op = Einsum::new(vec!["a".into(), "b".into()], "ij,jk->ik".into());
        op.execute(&vals, &mut out).unwrap();
        let r = out.floats().unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
        // = [58, 64, 139, 154]
        assert!((r[0] - 58.0).abs() < 1e-4, "r[0]={}", r[0]);
        assert!((r[1] - 64.0).abs() < 1e-4, "r[1]={}", r[1]);
        assert!((r[2] - 139.0).abs() < 1e-4, "r[2]={}", r[2]);
        assert!((r[3] - 154.0).abs() < 1e-4, "r[3]={}", r[3]);
    }

    #[test]
    fn bmc_bchw_bmhw() {
        // The actual equation used in the locate model.
        // A: (1,2,3)  B: (1,3,2,2) -> (1,2,2,2)
        // output[b,m,h,w] = sum_c A[b,m,c] * B[b,c,h,w]
        let a: Vec<f32> = (1..=6).map(|x| x as f32).collect(); // 1x2x3
        let b: Vec<f32> = (1..=12).map(|x| x as f32).collect(); // 1x3x2x2
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("a".into(), Tensor::new(dims![1, 2, 3], a));
        m.insert("b".into(), Tensor::new(dims![1, 3, 2, 2], b));
        let vals = Values { intermediates: &m, constants: Constants::empty() };
        let mut out = Tensor::new(dims![1, 2, 2, 2], vec![0.0f32; 8]);
        let mut op = Einsum::new(vec!["a".into(), "b".into()], "bmc,bchw->bmhw".into());
        op.execute(&vals, &mut out).unwrap();
        let r = out.floats().unwrap();
        // Manual: b=0, m=0: A[0,0,c]=[1,2,3]
        //   h=0,w=0: 1*B[0,0,0,0]+2*B[0,1,0,0]+3*B[0,2,0,0]= 1*1+2*5+3*9 = 1+10+27=38
        //   h=0,w=1: 1*2+2*6+3*10 = 2+12+30=44
        //   h=1,w=0: 1*3+2*7+3*11 = 3+14+33=50
        //   h=1,w=1: 1*4+2*8+3*12 = 4+16+36=56
        assert!((r[0] - 38.0).abs() < 1e-3, "r[0]={}", r[0]);
        assert!((r[1] - 44.0).abs() < 1e-3, "r[1]={}", r[1]);
        assert!((r[2] - 50.0).abs() < 1e-3, "r[2]={}", r[2]);
        assert!((r[3] - 56.0).abs() < 1e-3, "r[3]={}", r[3]);
    }
}
