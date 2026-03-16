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
        let mut starts = [0i64; 8];
        let mut ends = [0i64; 8];
        let mut steps = [1i64; 8];
        for ax in 0..rank {
            ends[ax] = input.dims[ax] as i64;
        }

        let starts_ints = starts_t.ints();
        let ends_ints = ends_t.ints();

        let mut axes_buf = [0usize; 8];
        let axes_len;
        if let Some(at) = axes_t {
            let at_ints = at.ints();
            axes_len = at_ints.len();
            for (i, &a) in at_ints.iter().enumerate() {
                axes_buf[i] = if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                };
            }
        } else {
            axes_len = starts_ints.len();
            for i in 0..axes_len {
                axes_buf[i] = i;
            }
        }

        for i in 0..axes_len {
            let ax = axes_buf[i];
            starts[ax] = starts_ints[i];
            ends[ax] = ends_ints[i];
            if let Some(st) = steps_t {
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

        let mut out_dims = [0usize; 8];
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

        let numel: usize = out_dims[..rank].iter().product();

        let mut in_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }
        let mut out_strides = [1usize; 8];
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
        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
