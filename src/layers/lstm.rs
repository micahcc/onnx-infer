use std::collections::HashMap;

use anyhow::Context;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::dims;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Clone, Copy, PartialEq)]
pub enum LstmDirection {
    Forward,
    Reverse,
    Bidirectional,
}

pub struct Lstm {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub hidden_size: usize,
    pub direction: LstmDirection,
}

impl Lstm {
    pub fn new(
        inputs: Vec<String>,
        outputs: Vec<String>,
        hidden_size: usize,
        direction: LstmDirection,
    ) -> Self {
        Self {
            inputs,
            outputs,
            hidden_size,
            direction,
        }
    }

    fn num_directions(&self) -> usize {
        if self.direction == LstmDirection::Bidirectional {
            2
        } else {
            1
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Run one direction of the LSTM.
/// X: [seq_len, batch, input_size]
/// W_dir: [4*hs, input_size] (one direction slice)
/// R_dir: [4*hs, hs]
/// Wb_dir: [4*hs], Rb_dir: [4*hs] (bias slices for one direction)
/// h0: [batch, hs], c0: [batch, hs]
/// y_out: output buffer [seq_len, batch, hs]
/// h_out: [batch, hs], c_out: [batch, hs]
fn run_lstm_direction(
    x: &[f32],
    seq_len: usize,
    batch: usize,
    input_size: usize,
    w: &[f32],
    r: &[f32],
    wb: &[f32],
    rb: &[f32],
    h0: &[f32],
    c0: &[f32],
    hs: usize,
    reverse: bool,
    y_out: &mut [f32],
    h_out: &mut [f32],
    c_out: &mut [f32],
) {
    let hs4 = 4 * hs;

    // Working buffers
    let mut h = vec![0.0f32; batch * hs];
    let mut c = vec![0.0f32; batch * hs];
    h.copy_from_slice(h0);
    c.copy_from_slice(c0);
    let mut gates = vec![0.0f32; batch * hs4];

    for step in 0..seq_len {
        let t = if reverse { seq_len - 1 - step } else { step };
        let x_t = &x[t * batch * input_size..(t + 1) * batch * input_size];

        // gates = X_t * W^T + H_{t-1} * R^T + Wb + Rb
        // Initialize gates with bias (Wb + Rb)
        for b in 0..batch {
            for g in 0..hs4 {
                gates[b * hs4 + g] = wb[g] + rb[g];
            }
        }

        // gates += X_t * W^T   (X_t: [batch, input_size], W: [4*hs, input_size])
        crate::blas::sgemm(
            batch, hs4, input_size, 1.0, x_t, input_size, false, w, input_size, true, 1.0,
            &mut gates, hs4,
        );

        // gates += H_{t-1} * R^T   (H: [batch, hs], R: [4*hs, hs])
        crate::blas::sgemm(
            batch, hs4, hs, 1.0, &h, hs, false, r, hs, true, 1.0, &mut gates, hs4,
        );

        // ONNX gate order: i, o, f, c (different from some frameworks)
        for b in 0..batch {
            let g = &gates[b * hs4..];
            let h_b = &mut h[b * hs..(b + 1) * hs];
            let c_b = &mut c[b * hs..(b + 1) * hs];
            for j in 0..hs {
                let i_gate = sigmoid(g[j]); // i: [0..hs)
                let o_gate = sigmoid(g[hs + j]); // o: [hs..2*hs)
                let f_gate = sigmoid(g[2 * hs + j]); // f: [2*hs..3*hs)
                let c_gate = g[3 * hs + j].tanh(); // c: [3*hs..4*hs)

                c_b[j] = f_gate * c_b[j] + i_gate * c_gate;
                h_b[j] = o_gate * c_b[j].tanh();
            }
        }

        // Write to Y output
        let y_slice = &mut y_out[t * batch * hs..(t + 1) * batch * hs];
        y_slice.copy_from_slice(&h);
    }

    h_out.copy_from_slice(&h);
    c_out.copy_from_slice(&c);
}

impl Layer for Lstm {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x_tensor = get_tensor(values, &self.inputs[0])?;
        let w_tensor = get_tensor(values, &self.inputs[1])?;
        let r_tensor = get_tensor(values, &self.inputs[2])?;

        let x = x_tensor.floats().context("in Lstm layer")?;
        let w = w_tensor.floats().context("in Lstm layer")?;
        let r = r_tensor.floats().context("in Lstm layer")?;

        let seq_len = x_tensor.dims[0];
        let batch = x_tensor.dims[1];
        let input_size = x_tensor.dims[2];
        let hs = self.hidden_size;
        let num_dir = self.num_directions();
        let hs4 = 4 * hs;

        // Bias: optional input[3], shape [num_dir, 8*hs]
        let zero_bias = vec![0.0f32; num_dir * 8 * hs];
        let bias = if self.inputs.len() > 3 && !self.inputs[3].is_empty() {
            get_tensor(values, &self.inputs[3])?
                .floats()
                .context("in Lstm layer")?
        } else {
            &zero_bias
        };

        // Initial hidden state: optional input[5], shape [num_dir, batch, hs]
        let zero_h = vec![0.0f32; num_dir * batch * hs];
        let init_h = if self.inputs.len() > 5 && !self.inputs[5].is_empty() {
            get_tensor(values, &self.inputs[5])?
                .floats()
                .context("in Lstm layer")?
        } else {
            &zero_h
        };

        // Initial cell state: optional input[6], shape [num_dir, batch, hs]
        let init_c = if self.inputs.len() > 6 && !self.inputs[6].is_empty() {
            get_tensor(values, &self.inputs[6])?
                .floats()
                .context("in Lstm layer")?
        } else {
            &zero_h
        };

        // Output Y: [seq_len, num_dir, batch, hs]
        let y_numel = seq_len * num_dir * batch * hs;
        let buf = output.as_mut_f32(y_numel);

        // Temp buffers for per-direction output
        let mut y_dir = vec![0.0f32; seq_len * batch * hs];
        let mut h_final = vec![0.0f32; batch * hs];
        let mut c_final = vec![0.0f32; batch * hs];

        for dir in 0..num_dir {
            let w_dir = &w[dir * hs4 * input_size..(dir + 1) * hs4 * input_size];
            let r_dir = &r[dir * hs4 * hs..(dir + 1) * hs4 * hs];
            let wb_dir = &bias[dir * 8 * hs..dir * 8 * hs + hs4];
            let rb_dir = &bias[dir * 8 * hs + hs4..dir * 8 * hs + 2 * hs4];
            let h0 = &init_h[dir * batch * hs..(dir + 1) * batch * hs];
            let c0 = &init_c[dir * batch * hs..(dir + 1) * batch * hs];

            let reverse = match self.direction {
                LstmDirection::Reverse => true,
                LstmDirection::Bidirectional => dir == 1,
                LstmDirection::Forward => false,
            };

            run_lstm_direction(
                x,
                seq_len,
                batch,
                input_size,
                w_dir,
                r_dir,
                wb_dir,
                rb_dir,
                h0,
                c0,
                hs,
                reverse,
                &mut y_dir,
                &mut h_final,
                &mut c_final,
            );

            // Interleave into Y: [seq_len, num_dir, batch, hs]
            for t in 0..seq_len {
                for b in 0..batch {
                    let src_off = t * batch * hs + b * hs;
                    let dst_off = t * num_dir * batch * hs + dir * batch * hs + b * hs;
                    buf[dst_off..dst_off + hs].copy_from_slice(&y_dir[src_off..src_off + hs]);
                }
            }
        }

        let out_dims: Dims = dims![seq_len, num_dir, batch, hs];
        output.set_dims(&out_dims);
        Ok(())
    }
}
