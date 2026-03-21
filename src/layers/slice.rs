use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

fn read_i64_into(t: &Tensor, buf: &mut [i64; 8]) -> anyhow::Result<usize> {
    let len = t.numel();
    match t.dtype() {
        DType::Int64 => {
            for (i, &v) in t.ints().context("in Slice layer")?.iter().enumerate() {
                buf[i] = v;
            }
        }
        DType::Float => {
            for (i, &v) in t.floats().context("in Slice layer")?.iter().enumerate() {
                buf[i] = v as i64;
            }
        }
        DType::String => anyhow::bail!("strings not supported in Slice"),
    }
    Ok(len)
}

pub struct Slice {
    pub inputs: Vec<String>,
    // For opset < 10: starts/ends/axes as attributes
    pub attr_starts: Option<Vec<i64>>,
    pub attr_ends: Option<Vec<i64>>,
    pub attr_axes: Option<Vec<i64>>,
}

impl Slice {
    pub fn new(inputs: Vec<String>) -> Self {
        Self {
            inputs,
            attr_starts: None,
            attr_ends: None,
            attr_axes: None,
        }
    }

    pub fn new_v1(
        inputs: Vec<String>,
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
    ) -> Self {
        Self {
            inputs,
            attr_starts: Some(starts),
            attr_ends: Some(ends),
            attr_axes: axes,
        }
    }
}

impl Layer for Slice {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();
        let mut starts = [0i64; 8];
        let mut ends = [0i64; 8];
        let mut steps = [1i64; 8];
        for (ax, e) in ends.iter_mut().enumerate().take(rank) {
            *e = input.dims[ax] as i64;
        }

        if let Some(attr_starts) = &self.attr_starts {
            // Opset < 10: attribute-based slice
            let attr_ends = self.attr_ends.as_ref().unwrap();
            let starts_len = attr_starts.len();
            let mut starts_buf = [0i64; 8];
            let mut ends_buf = [0i64; 8];
            for (i, &v) in attr_starts.iter().enumerate() {
                starts_buf[i] = v;
            }
            for (i, &v) in attr_ends.iter().enumerate() {
                ends_buf[i] = v;
            }
            let mut axes_buf = [0usize; 8];
            let axes_len;
            if let Some(attr_axes) = &self.attr_axes {
                axes_len = attr_axes.len();
                for (i, &v) in attr_axes.iter().enumerate() {
                    axes_buf[i] = if v < 0 {
                        (rank as i64 + v) as usize
                    } else {
                        v as usize
                    };
                }
            } else {
                axes_len = starts_len;
                for (i, ab) in axes_buf.iter_mut().enumerate().take(axes_len) {
                    *ab = i;
                }
            }
            for i in 0..axes_len {
                let ax = axes_buf[i];
                starts[ax] = starts_buf[i];
                ends[ax] = ends_buf[i];
            }
        } else {
            // Opset >= 10: input-based slice
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

            let mut starts_buf = [0i64; 8];
            let starts_len = read_i64_into(starts_t, &mut starts_buf)?;
            let mut ends_buf = [0i64; 8];
            read_i64_into(ends_t, &mut ends_buf)?;

            let mut axes_buf = [0usize; 8];
            let axes_len;
            if let Some(at) = axes_t {
                let mut at_buf = [0i64; 8];
                axes_len = read_i64_into(at, &mut at_buf)?;
                for i in 0..axes_len {
                    axes_buf[i] = if at_buf[i] < 0 {
                        (rank as i64 + at_buf[i]) as usize
                    } else {
                        at_buf[i] as usize
                    };
                }
            } else {
                axes_len = starts_len;
                for (i, ab) in axes_buf.iter_mut().enumerate().take(axes_len) {
                    *ab = i;
                }
            }

            let mut steps_buf = [1i64; 8];
            if let Some(st) = steps_t {
                read_i64_into(st, &mut steps_buf)?;
            }

            for i in 0..axes_len {
                let ax = axes_buf[i];
                starts[ax] = starts_buf[i];
                ends[ax] = ends_buf[i];
                steps[ax] = steps_buf[i];
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

        #[allow(clippy::needless_range_loop)]
        match input.dtype() {
            DType::Float => {
                let in_data = input.floats().context("in Slice layer")?;
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
                let in_data = input.ints().context("in Slice layer")?;
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
            DType::String => unreachable!("strings not supported"),
        }
        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
