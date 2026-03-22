use std::collections::HashMap;

use anyhow::Context;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

/// Resize operator implementation
///
/// The Resize operator in ONNX has evolved across different opset versions, leading to
/// various input parameter formats:
///
/// 1. Explicit Output Sizes (ONNX opset 11+)
///    ```rust,ignore
///    Resize[mode=nearest]
///      Input[0]: data[1, 64, 26, 26]      # Input tensor to resize
///      Input[1]: roi[] (optional)         # Region of interest (usually empty)
///      Input[2]: scales[] (empty)         # Scale factors (ignored when sizes provided)
///      Input[3]: sizes[1, 64, 52, 52]     # Explicit output dimensions
///    ```
///
/// 2. Scale Factors in Third Input (ONNX opset 11)
///    ```rust,ignore
///    Resize[mode=nearest]
///      Input[0]: data[1, 64, 26, 26]      # Input tensor to resize
///      Input[1]: roi[] (optional)         # Region of interest (usually empty)
///      Input[2]: scales[1.0, 1.0, 2.0, 2.0] # Scale factors for each dimension
///      Input[3]: sizes[] (empty)          # Not provided when using scales
///    ```
///
/// 3. Scale Factors in Second Input (ONNX opset 10 or YOLOv4-style)
///    ```rust,ignore
///    Resize[mode=nearest]
///      Input[0]: data[1, 64, 26, 26]      # Input tensor to resize
///      Input[1]: scales[1.0, 1.0, 2.0, 2.0] # Scale factors
///    ```
///    This format is common in YOLOv4 models converted from Darknet.
///
#[derive(Debug)]
pub struct Resize {
    pub inputs: Vec<String>,
    pub mode: ResizeMode,
    pub coord_transform: CoordTransform,
    pub nearest_mode: NearestMode,
    pub nhwc: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Debug, Clone, Copy)]
pub enum CoordTransform {
    HalfPixel,
    PyTorchHalfPixel,
    Asymmetric,
    AlignCorners,
}

#[derive(Debug, Clone, Copy)]
pub enum NearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

impl Resize {
    pub fn new(inputs: Vec<String>, mode: &str, coord_transform: &str, nearest_mode: &str, nhwc: bool) -> Self {
        let mode = match mode {
            "linear" => ResizeMode::Linear,
            "cubic" => ResizeMode::Cubic,
            _ => ResizeMode::Nearest,
        };
        let coord_transform = match coord_transform {
            "asymmetric" => CoordTransform::Asymmetric,
            "align_corners" => CoordTransform::AlignCorners,
            "pytorch_half_pixel" => CoordTransform::PyTorchHalfPixel,
            _ => CoordTransform::HalfPixel,
        };
        let nearest_mode = match nearest_mode {
            "floor" => NearestMode::Floor,
            "ceil" => NearestMode::Ceil,
            "round_prefer_floor" => NearestMode::RoundPreferFloor,
            _ => NearestMode::RoundPreferCeil,
        };
        Self {
            inputs,
            mode,
            coord_transform,
            nearest_mode,
            nhwc,
        }
    }
}

impl Resize {
    /// Map an output coordinate to a floating-point input coordinate
    fn map_coord(&self, out_coord: usize, out_dim: usize, in_dim: usize, scale: f32) -> f32 {
        match self.coord_transform {
            CoordTransform::HalfPixel => (out_coord as f32 + 0.5) / scale - 0.5,
            CoordTransform::PyTorchHalfPixel => {
                if out_dim <= 1 {
                    0.0
                } else {
                    (out_coord as f32 + 0.5) / scale - 0.5
                }
            }
            CoordTransform::Asymmetric => out_coord as f32 / scale,
            CoordTransform::AlignCorners => {
                if out_dim <= 1 {
                    0.0
                } else {
                    out_coord as f32 * (in_dim - 1) as f32 / (out_dim - 1) as f32
                }
            }
        }
    }
}

impl Layer for Resize {
    /// Execute the Resize operation
    ///
    /// This implementation supports multiple ONNX formats:
    /// - Explicit output sizes (opset 11+)
    /// - Scale factors in 3rd input (opset 11)
    /// - Scale factors in 2nd input (opset 10, YOLOv4)
    /// - Fallback for YOLOv4 models with missing scale data
    ///
    /// The implementation first determines the output dimensions based on available inputs,
    /// then performs nearest-neighbor interpolation using the specified coordinate transformation
    /// mode and rounding method.
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();

        // When graph_opt converts to NHWC, only the data input is transposed.
        // Scales/sizes remain in NCHW axis order [N,C,H,W].
        // Remap to NHWC order [N,H,W,C] so axis indices match the data layout.
        let remap_nhwc = |dims: &mut [usize; 8]| {
            if self.nhwc && rank == 4 {
                // NCHW [N,C,H,W] → NHWC [N,H,W,C]
                let c = dims[1];
                dims[1] = dims[2];
                dims[2] = dims[3];
                dims[3] = c;
            }
        };
        let remap_scales_nhwc = |scales: &[f32]| -> [f32; 4] {
            if self.nhwc && scales.len() == 4 {
                [scales[0], scales[2], scales[3], scales[1]]
            } else {
                let mut out = [1.0f32; 4];
                for (i, &s) in scales.iter().enumerate().take(4) {
                    out[i] = s;
                }
                out
            }
        };

        let mut out_dims = [0usize; 8];

        // Case 1: Output sizes are explicitly provided (ONNX opset 11+)
        // Example:
        //   Resize[mode=nearest]
        //     Input: data[1, 64, 26, 26]
        //     Input: roi (optional, often empty)
        //     Input: scales (empty or ignored)
        //     Input: sizes[1, 64, 52, 52] (explicit target dimensions)
        // Result: output shape will be exactly [1, 64, 52, 52]
        if self.inputs.len() > 3 && !self.inputs[3].is_empty() {
            let sizes = get_tensor(values, &self.inputs[3])?;
            for (i, &v) in sizes.ints().context("in Resize layer")?.iter().enumerate() {
                out_dims[i] = v as usize;
            }
            // sizes are in NCHW axis order; remap to NHWC if needed
            remap_nhwc(&mut out_dims);
        }
        // Case 2: Scales are provided in the third input (ONNX opset 11)
        else if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            let scales = get_tensor(values, &self.inputs[2])?;
            let scales_f = scales.floats().context("in Resize layer")?;
            let remapped = remap_scales_nhwc(scales_f);
            for (i, (&d, &s)) in input.dims.iter().zip(remapped.iter()).enumerate() {
                out_dims[i] = (d as f32 * s) as usize;
            }
        }
        // Case 3: Scales are provided in the second input (ONNX opset 10 / YOLOv4)
        else if self.inputs.len() > 1 && !self.inputs[1].is_empty() {
            let scales = get_tensor(values, &self.inputs[1])?;
            if !scales.floats().context("in Resize layer")?.is_empty() {
                let scales_f = scales.floats().context("in Resize layer")?;
                let remapped = remap_scales_nhwc(scales_f);
                for (i, (&d, &s)) in input.dims.iter().zip(remapped.iter()).enumerate() {
                    out_dims[i] = (d as f32 * s) as usize;
                }
            }
            // Case 3b: Fallback — 2x scaling of spatial dimensions
            else {
                for (i, &d) in input.dims.iter().enumerate() {
                    let is_spatial = if self.nhwc { i >= 1 && i < rank - 1 } else { i >= 2 };
                    let scale = if is_spatial { 2.0 } else { 1.0 };
                    out_dims[i] = (d as f32 * scale) as usize;
                }
            }
        }
        // Error case: No valid sizing information provided
        else {
            anyhow::bail!("Resize: no scales or sizes");
        }

        let numel: usize = out_dims[..rank].iter().product();

        let mut in_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        let input_f = input.floats().context("in Resize layer")?;
        let buf = output.as_mut_f32(numel);

        assert!(self.nhwc, "Resize::execute requires NHWC input layout");

        if self.mode == ResizeMode::Linear && rank == 4 {
            // Bilinear interpolation for 4D tensors (NHWC: [N, H, W, C])
            let (n, c, oh, ow, ih, iw) =
                (out_dims[0], out_dims[3], out_dims[1], out_dims[2], input.dims[1], input.dims[2]);
            let h_scale = oh as f32 / ih as f32;
            let w_scale = ow as f32 / iw as f32;

            let h_coords: Vec<(usize, usize, f32)> = (0..oh)
                .map(|y| {
                    let orig = self.map_coord(y, oh, ih, h_scale)
                        .max(0.0)
                        .min((ih - 1) as f32);
                    let y0 = orig.floor() as usize;
                    let y1 = (y0 + 1).min(ih - 1);
                    let fy = orig - orig.floor();
                    (y0, y1, fy)
                })
                .collect();
            let w_coords: Vec<(usize, usize, f32)> = (0..ow)
                .map(|x| {
                    let orig = self.map_coord(x, ow, iw, w_scale)
                        .max(0.0)
                        .min((iw - 1) as f32);
                    let x0 = orig.floor() as usize;
                    let x1 = (x0 + 1).min(iw - 1);
                    let fx = orig - orig.floor();
                    (x0, x1, fx)
                })
                .collect();

            // NHWC: output[b][y][x][ch]
            let mut out_idx = 0;
            for b in 0..n {
                let b_off = b * ih * iw * c;
                for y in 0..oh {
                    let (y0, y1, fy) = h_coords[y];
                    for x in 0..ow {
                        let (x0, x1, fx) = w_coords[x];
                        let off00 = b_off + (y0 * iw + x0) * c;
                        let off01 = b_off + (y0 * iw + x1) * c;
                        let off10 = b_off + (y1 * iw + x0) * c;
                        let off11 = b_off + (y1 * iw + x1) * c;
                        for ch in 0..c {
                            buf[out_idx] = input_f[off00 + ch] * (1.0 - fy) * (1.0 - fx)
                                + input_f[off01 + ch] * (1.0 - fy) * fx
                                + input_f[off10 + ch] * fy * (1.0 - fx)
                                + input_f[off11 + ch] * fy * fx;
                            out_idx += 1;
                        }
                    }
                }
            }
        } else {
            // Nearest-neighbor interpolation (original path)
            let mut coord_tables: [Vec<usize>; 8] = Default::default();
            for ax in 0..rank {
                let scale = out_dims[ax] as f32 / input.dims[ax] as f32;
                let max_in = (input.dims[ax] - 1) as f32;
                coord_tables[ax] = (0..out_dims[ax])
                    .map(|out_coord| {
                        let orig = self.map_coord(out_coord, out_dims[ax], input.dims[ax], scale);
                        let in_coord = match self.nearest_mode {
                            NearestMode::RoundPreferCeil => orig.round(),
                            NearestMode::RoundPreferFloor => {
                                if orig - orig.floor() == 0.5 {
                                    orig.floor()
                                } else {
                                    orig.round()
                                }
                            }
                            NearestMode::Floor => orig.floor(),
                            NearestMode::Ceil => orig.ceil(),
                        };
                        in_coord.max(0.0).min(max_in) as usize * in_strides[ax]
                    })
                    .collect();
            }

            let mut coord = [0usize; 8];
            let mut in_off = 0usize;
            for table in coord_tables.iter().take(rank) {
                in_off += table[0];
            }
            #[allow(clippy::needless_range_loop)]
            for out_flat in 0..numel {
                buf[out_flat] = input_f[in_off];

                for d in (0..rank).rev() {
                    let old = coord[d];
                    coord[d] += 1;
                    if coord[d] < out_dims[d] {
                        in_off = in_off - coord_tables[d][old] + coord_tables[d][coord[d]];
                        break;
                    }
                    in_off -= coord_tables[d][old];
                    coord[d] = 0;
                    in_off += coord_tables[d][0];
                }
            }
        }

        output.set_dims(&out_dims[..rank]);
        if self.nhwc {
            output.layout = crate::Layout::NHWC;
        }
        Ok(())
    }
}
