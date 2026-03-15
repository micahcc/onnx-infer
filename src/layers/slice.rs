use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Slice {
    pub inputs: Vec<String>,
}

impl Slice {
    pub fn new(inputs: Vec<String>) -> Self {
        Self { inputs }
    }
}

impl Layer for Slice {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let starts_t = get_tensor(values, &self.inputs[1])?;
        let ends_t = get_tensor(values, &self.inputs[2])?;
        let axes_t = if self.inputs.len() > 3 && !self.inputs[3].is_empty() {
            Some(get_tensor(values, &self.inputs[3])?)
        } else {
            None
        };
        let steps_t = if self.inputs.len() > 4 && !self.inputs[4].is_empty() {
            Some(get_tensor(values, &self.inputs[4])?)
        } else {
            None
        };

        let rank = input.dims.len();
        let mut starts = vec![0i64; rank];
        let mut ends: Vec<i64> = input.dims.iter().map(|&d| d as i64).collect();
        let mut steps = vec![1i64; rank];

        let starts_ints = starts_t.ints();
        let ends_ints = ends_t.ints();

        let axes: Vec<usize> = if let Some(ref at) = axes_t {
            at.ints()
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..starts_ints.len()).collect()
        };

        for (i, &ax) in axes.iter().enumerate() {
            starts[ax] = starts_ints[i];
            ends[ax] = ends_ints[i];
            if let Some(ref st) = steps_t {
                steps[ax] = st.ints()[i];
            }
        }

        for ax in 0..rank {
            let dim = input.dims[ax] as i64;
            if steps[ax] > 0 {
                if starts[ax] < 0 {
                    starts[ax] += dim;
                }
                if ends[ax] < 0 {
                    ends[ax] += dim;
                }
                starts[ax] = starts[ax].clamp(0, dim);
                ends[ax] = ends[ax].clamp(0, dim);
            } else {
                if starts[ax] < 0 {
                    starts[ax] += dim;
                }
                if ends[ax] < -dim {
                    ends[ax] = -1;
                } else if ends[ax] < 0 {
                    ends[ax] += dim;
                }
                starts[ax] = starts[ax].clamp(-1, dim - 1);
                ends[ax] = ends[ax].clamp(-1, dim - 1);
            }
        }

        let mut out_dims = vec![0usize; rank];
        for ax in 0..rank {
            if steps[ax] > 0 {
                if ends[ax] <= starts[ax] {
                    out_dims[ax] = 0;
                } else {
                    out_dims[ax] = ((ends[ax] - starts[ax] - 1) / steps[ax] + 1) as usize;
                }
            } else if starts[ax] <= ends[ax] {
                out_dims[ax] = 0;
            } else {
                out_dims[ax] = ((starts[ax] - ends[ax] - 1) / (-steps[ax]) + 1) as usize;
            }
        }

        let numel: usize = out_dims.iter().product();

        let mut in_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }

        match input.dtype() {
            DType::Float => {
                let in_data = input.floats();
                let buf = output.as_mut_f32(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for ax in 0..rank {
                        let coord = remaining / out_strides[ax];
                        remaining %= out_strides[ax];
                        let in_coord = starts[ax] + coord as i64 * steps[ax];
                        in_flat += in_coord as usize * in_strides[ax];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
            DType::Int64 => {
                let in_data = input.ints();
                let buf = output.as_mut_i64(numel);
                for out_flat in 0..numel {
                    let mut remaining = out_flat;
                    let mut in_flat = 0;
                    for ax in 0..rank {
                        let coord = remaining / out_strides[ax];
                        remaining %= out_strides[ax];
                        let in_coord = starts[ax] + coord as i64 * steps[ax];
                        in_flat += in_coord as usize * in_strides[ax];
                    }
                    buf[out_flat] = in_data[in_flat];
                }
            }
        }
        output.dims = out_dims;
        Ok(())
    }
}
