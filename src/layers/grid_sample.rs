//! GridSample operator: sample an input tensor using a flow field.
//!
//! ONNX spec: <https://onnx.ai/onnx/operators/onnx__GridSample.html>
//!
//! Input 0: `(N, C, H_in, W_in)` — the feature map to sample from.
//! Input 1: `(N, H_out, W_out, 2)` — normalised sampling coordinates in `[-1, 1]`.
//! Output:  `(N, C, H_out, W_out)`.

use anyhow::bail;

use crate::Result;
use crate::Tensor;
use crate::Values;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridSampleMode {
    Bilinear,
    Nearest,
    Bicubic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    Zeros,
    Border,
    Reflection,
}

#[derive(Debug)]
pub struct GridSample {
    inputs: Vec<String>,
    mode: GridSampleMode,
    padding_mode: PaddingMode,
    align_corners: bool,
}

impl GridSample {
    pub fn new(
        inputs: Vec<String>,
        mode: GridSampleMode,
        padding_mode: PaddingMode,
        align_corners: bool,
    ) -> Self {
        Self { inputs, mode, padding_mode, align_corners }
    }
}

/// Map a normalised coordinate to pixel space.
fn unnormalise(coord: f32, size: usize, align_corners: bool) -> f32 {
    if align_corners {
        (coord + 1.0) * 0.5 * (size as f32 - 1.0)
    } else {
        (coord + 1.0) * 0.5 * size as f32 - 0.5
    }
}

/// Apply padding to a raw pixel coordinate, returning `None` when out-of-bounds
/// under `Zeros` mode.
fn apply_padding(x: f32, size: usize, mode: PaddingMode) -> Option<f32> {
    let n = size as f32;
    match mode {
        PaddingMode::Zeros => {
            if x < -0.5 || x > n - 0.5 { None } else { Some(x) }
        }
        PaddingMode::Border => Some(x.clamp(0.0, n - 1.0)),
        PaddingMode::Reflection => {
            // Reflect about 0 and (n-1).
            let period = if size > 1 { 2.0 * (n - 1.0) } else { 1.0 };
            let mut v = x.abs() % period;
            if v > n - 1.0 { v = period - v; }
            Some(v.clamp(0.0, n - 1.0))
        }
    }
}

#[allow(dead_code)]
fn fetch(
    data: &[f32],
    n: usize, ch: usize,
    h_in: usize, w_in: usize,
    yi: i64, xi: i64,
    padding_mode: PaddingMode,
) -> f32 {
    let xf = apply_padding(xi as f32, w_in, padding_mode);
    let yf = apply_padding(yi as f32, h_in, padding_mode);
    match (xf, yf) {
        (Some(xf), Some(yf)) => {
            let xi = xf.round().clamp(0.0, w_in as f32 - 1.0) as usize;
            let yi = yf.round().clamp(0.0, h_in as f32 - 1.0) as usize;
            data[n * ch * h_in * w_in + ch * h_in * w_in + yi * w_in + xi]
        }
        _ => 0.0,
    }
}

/// Keys cubic kernel weights for offset `|t|`.
fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.75f32;
    let t = t.abs();
    let t2 = t * t;
    let t3 = t2 * t;
    [
        a * t3 - 2.0 * a * t2 + a * t,
        (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
        -(a + 2.0) * t3 + (a + 3.0) * t2,
        a * t3 - a * t2,
    ]
}

impl Layer for GridSample {
    fn execute(&mut self, values: &Values, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let grid  = get_tensor(values, &self.inputs[1])?;

        let is = &input.dims;
        let gs = &grid.dims;
        if is.len() != 4 { bail!("GridSample: input must be 4D (N,C,H,W), got {}D", is.len()); }
        if gs.len() != 4 { bail!("GridSample: grid must be 4D (N,H,W,2), got {}D", gs.len()); }

        let (n_batch, c, h_in, w_in) = (is[0], is[1], is[2], is[3]);
        let (_, h_out, w_out, _) = (gs[0], gs[1], gs[2], gs[3]);

        let numel = n_batch * c * h_out * w_out;
        let in_data: Vec<f32> = input.floats()?.to_vec();
        let gr_data: Vec<f32> = grid.floats()?.to_vec();
        let out_data = output.as_mut_f32(numel);

        let align = self.align_corners;
        let pad   = self.padding_mode;

        for n in 0..n_batch {
            for ch in 0..c {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let g = n * h_out * w_out * 2 + ho * w_out * 2 + wo * 2;
                        let gx = gr_data[g];
                        let gy = gr_data[g + 1];

                        let x = unnormalise(gx, w_in, align);
                        let y = unnormalise(gy, h_in, align);

                        let fetch_px = |yi: i64, xi: i64| -> f32 {
                            let xf = apply_padding(xi as f32, w_in, pad);
                            let yf = apply_padding(yi as f32, h_in, pad);
                            match (xf, yf) {
                                (Some(xf), Some(yf)) => {
                                    let xi = xf.round().clamp(0.0, w_in as f32 - 1.0) as usize;
                                    let yi = yf.round().clamp(0.0, h_in as f32 - 1.0) as usize;
                                    in_data[n * c * h_in * w_in + ch * h_in * w_in + yi * w_in + xi]
                                }
                                _ => 0.0,
                            }
                        };

                        let val = match self.mode {
                            GridSampleMode::Nearest => {
                                fetch_px(y.round() as i64, x.round() as i64)
                            }
                            GridSampleMode::Bilinear => {
                                let x0 = x.floor() as i64;
                                let y0 = y.floor() as i64;
                                let dx = x - x0 as f32;
                                let dy = y - y0 as f32;
                                fetch_px(y0,     x0    ) * (1.0 - dx) * (1.0 - dy)
                                    + fetch_px(y0,     x0 + 1) * dx * (1.0 - dy)
                                    + fetch_px(y0 + 1, x0    ) * (1.0 - dx) * dy
                                    + fetch_px(y0 + 1, x0 + 1) * dx * dy
                            }
                            GridSampleMode::Bicubic => {
                                let xf = x.floor() as i64;
                                let yf = y.floor() as i64;
                                let wx = cubic_weights(x - xf as f32);
                                let wy = cubic_weights(y - yf as f32);
                                let mut acc = 0.0f32;
                                for r in 0..4i64 {
                                    for col in 0..4i64 {
                                        acc += wy[r as usize]
                                            * wx[col as usize]
                                            * fetch_px(yf + r - 1, xf + col - 1);
                                    }
                                }
                                acc
                            }
                        };

                        out_data[n * c * h_out * w_out + ch * h_out * w_out + ho * w_out + wo] = val;
                    }
                }
            }
        }
        output.set_dims(&[n_batch, c, h_out, w_out]);
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

    fn make_vals<'a>(
        map: &'a HashMap<String, Tensor>,
    ) -> Values<'a> {
        Values { intermediates: map, constants: Constants::empty() }
    }

    /// 2×2 input with identity grid (every output pixel maps to itself).
    #[test]
    fn identity_grid_bilinear() {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("x".into(), Tensor::new(dims![1, 1, 2, 2], vec![1.0f32, 2.0, 3.0, 4.0]));
        // Grid shape: (N=1, H_out=2, W_out=2, 2)
        // With align_corners=true: (-1,-1)→(0,0), (1,-1)→(1,0), (-1,1)→(0,1), (1,1)→(1,1)
        let grid = vec![
            -1.0f32, -1.0,
             1.0,    -1.0,
            -1.0,     1.0,
             1.0,     1.0,
        ];
        map.insert("g".into(), Tensor::new(dims![1, 2, 2, 2], grid));

        let mut out = Tensor::new(dims![1, 1, 2, 2], vec![0.0f32; 4]);
        let mut op = GridSample::new(
            vec!["x".into(), "g".into()],
            GridSampleMode::Bilinear,
            PaddingMode::Zeros,
            true,
        );
        op.execute(&make_vals(&map), &mut out).unwrap();
        let got = out.floats().unwrap();
        assert!((got[0] - 1.0).abs() < 1e-5, "got {}", got[0]);
        assert!((got[1] - 2.0).abs() < 1e-5, "got {}", got[1]);
        assert!((got[2] - 3.0).abs() < 1e-5, "got {}", got[2]);
        assert!((got[3] - 4.0).abs() < 1e-5, "got {}", got[3]);
    }

    /// Out-of-bounds coordinates return 0 under Zeros padding.
    ///
    /// Grid coord (3.0, 3.0) with align_corners=true on a 2×2 image
    /// unnormalises to pixel (2.0, 2.0). Both bilinear floor neighbors
    /// (x0=2, x0+1=3) exceed the valid range [0, 1], so all four
    /// interpolation weights sample out-of-bounds → result is 0.
    #[test]
    fn out_of_bounds_zeros_padding() {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("x".into(), Tensor::new(dims![1, 1, 2, 2], vec![1.0f32, 2.0, 3.0, 4.0]));
        map.insert("g".into(), Tensor::new(dims![1, 1, 1, 2], vec![3.0f32, 3.0]));

        let mut out = Tensor::new(dims![1, 1, 1, 1], vec![0.0f32]);
        let mut op = GridSample::new(
            vec!["x".into(), "g".into()],
            GridSampleMode::Bilinear,
            PaddingMode::Zeros,
            true,
        );
        op.execute(&make_vals(&map), &mut out).unwrap();
        assert!((out.floats().unwrap()[0]).abs() < 1e-5);
    }

    /// Nearest mode: centre pixel of a 3×3 grid.
    #[test]
    fn nearest_mode() {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("x".into(), Tensor::new(dims![1, 1, 3, 3],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
        // (0,0) in normalised coords with align_corners=true → centre pixel (1,1) → value 5
        map.insert("g".into(), Tensor::new(dims![1, 1, 1, 2], vec![0.0f32, 0.0]));

        let mut out = Tensor::new(dims![1, 1, 1, 1], vec![0.0f32]);
        let mut op = GridSample::new(
            vec!["x".into(), "g".into()],
            GridSampleMode::Nearest,
            PaddingMode::Zeros,
            true,
        );
        op.execute(&make_vals(&map), &mut out).unwrap();
        assert!((out.floats().unwrap()[0] - 5.0).abs() < 1e-5);
    }
}
