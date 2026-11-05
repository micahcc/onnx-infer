use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct GatherElements {
    inputs: Vec<String>,
    axis: i64,
}

impl GatherElements {
    pub fn new(inputs: Vec<String>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl Layer for GatherElements {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let data = get_tensor(values, &self.inputs[0])?;
        let indices = get_tensor(values, &self.inputs[1])?;

        let rank = data.dims.len();
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Accept float indices (e.g. from ArgMax or Cast output in float form) by
        // converting to i64 via truncation.
        let idx_data_owned: Vec<i64>;
        let idx_data: &[i64] = if let Ok(v) = indices.ints() {
            v
        } else if let Ok(v) = indices.floats() {
            idx_data_owned = v.iter().map(|&x| x as i64).collect();
            &idx_data_owned
        } else {
            anyhow::bail!("GatherElements: indices must be int64 or float");
        };
        let numel = indices.numel();

        // Strides for indexing into data using indices' shape
        let mut idx_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            idx_strides[i] = idx_strides[i + 1] * indices.dims[i + 1];
        }
        let mut data_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data.dims[i + 1];
        }

        match data.dtype() {
            DType::Float => {
                let src = data.floats().context("GatherElements: data")?;
                let buf = output.as_mut_f32(numel);
                for flat in 0..numel {
                    // Decompose flat index into multi-dim coords in indices shape
                    let mut rem = flat;
                    let mut data_flat = 0usize;
                    for d in 0..rank {
                        let coord = rem / idx_strides[d];
                        rem %= idx_strides[d];
                        if d == axis {
                            let raw = idx_data[flat];
                            let gather_coord = if raw < 0 {
                                (data.dims[d] as i64 + raw) as usize
                            } else {
                                raw as usize
                            };
                            data_flat += gather_coord * data_strides[d];
                        } else {
                            data_flat += coord * data_strides[d];
                        }
                    }
                    buf[flat] = src[data_flat];
                }
            }
            DType::Int64 => {
                let src = data.ints().context("GatherElements: data")?;
                let buf = output.as_mut_i64(numel);
                for flat in 0..numel {
                    let mut rem = flat;
                    let mut data_flat = 0usize;
                    for d in 0..rank {
                        let coord = rem / idx_strides[d];
                        rem %= idx_strides[d];
                        if d == axis {
                            let raw = idx_data[flat];
                            let gather_coord = if raw < 0 {
                                (data.dims[d] as i64 + raw) as usize
                            } else {
                                raw as usize
                            };
                            data_flat += gather_coord * data_strides[d];
                        } else {
                            data_flat += coord * data_strides[d];
                        }
                    }
                    buf[flat] = src[data_flat];
                }
            }
            DType::String => unreachable!(),
        }
        output.set_dims(&indices.dims.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::Constants;
    use crate::Tensor;
    use crate::Values;
    use crate::dims;

    #[test]
    fn gather_along_axis0() {
        // data = [[1,2],[3,4],[5,6]], indices = [[0,1],[2,0]], axis=0
        // output[i,j] = data[indices[i,j], j]
        // output[0,0]=data[0,0]=1, output[0,1]=data[1,1]=4
        // output[1,0]=data[2,0]=5, output[1,1]=data[0,1]=2
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "data".into(),
            Tensor::new(dims![3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        m.insert(
            "idx".into(),
            Tensor::new_i64(dims![2, 2], vec![0i64, 1, 2, 0]),
        );
        let vals = Values {
            intermediates: &m,
            constants: Constants::empty(),
        };
        let mut out = Tensor::new(dims![2, 2], vec![0.0f32; 4]);
        let mut op = GatherElements::new(vec!["data".into(), "idx".into()], 0);
        op.execute(&vals, &mut out).unwrap();
        let r = out.floats().unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5, "r[0]={}", r[0]);
        assert!((r[1] - 4.0).abs() < 1e-5, "r[1]={}", r[1]);
        assert!((r[2] - 5.0).abs() < 1e-5, "r[2]={}", r[2]);
        assert!((r[3] - 2.0).abs() < 1e-5, "r[3]={}", r[3]);
    }

    #[test]
    fn gather_along_axis1() {
        // data = [[1,2,3],[4,5,6]], indices = [[2,0],[1,2]], axis=1
        // output[i,j] = data[i, indices[i,j]]
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "data".into(),
            Tensor::new(dims![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        m.insert(
            "idx".into(),
            Tensor::new_i64(dims![2, 2], vec![2i64, 0, 1, 2]),
        );
        let vals = Values {
            intermediates: &m,
            constants: Constants::empty(),
        };
        let mut out = Tensor::new(dims![2, 2], vec![0.0f32; 4]);
        let mut op = GatherElements::new(vec!["data".into(), "idx".into()], 1);
        op.execute(&vals, &mut out).unwrap();
        let r = out.floats().unwrap();
        assert!((r[0] - 3.0).abs() < 1e-5); // data[0,2]
        assert!((r[1] - 1.0).abs() < 1e-5); // data[0,0]
        assert!((r[2] - 5.0).abs() < 1e-5); // data[1,1]
        assert!((r[3] - 6.0).abs() < 1e-5); // data[1,2]
    }
}
