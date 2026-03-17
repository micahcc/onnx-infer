use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Resize {
    pub inputs: Vec<String>,
    pub coord_transform: CoordTransform,
    pub nearest_mode: NearestMode,
}

#[derive(Clone, Copy)]
pub enum CoordTransform {
    HalfPixel,
    Asymmetric,
    AlignCorners,
}

#[derive(Clone, Copy)]
pub enum NearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

impl Resize {
    pub fn new(inputs: Vec<String>, coord_transform: &str, nearest_mode: &str) -> Self {
        let coord_transform = match coord_transform {
            "asymmetric" => CoordTransform::Asymmetric,
            "align_corners" => CoordTransform::AlignCorners,
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
            coord_transform,
            nearest_mode,
        }
    }
}

impl Layer for Resize {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let input = get_tensor(values, &self.inputs[0])?;
        let rank = input.dims.len();

        let mut out_dims = [0usize; 8];
        if self.inputs.len() > 3 && !self.inputs[3].is_empty() {
            let sizes = get_tensor(values, &self.inputs[3])?;
            for (i, &v) in sizes.ints().iter().enumerate() {
                out_dims[i] = v as usize;
            }
        } else if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            let scales = get_tensor(values, &self.inputs[2])?;
            let scales_f = scales.floats();
            for (i, (&d, &s)) in input.dims.iter().zip(scales_f.iter()).enumerate() {
                out_dims[i] = (d as f32 * s) as usize;
            }
        } else {
            return Err(InferenceError::InvalidModel(
                "Resize: no scales or sizes".into(),
            ));
        }

        let numel: usize = out_dims[..rank].iter().product();

        let mut out_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        }
        let mut in_strides = [1usize; 8];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.dims[i + 1];
        }

        // Precompute per-axis coordinate lookup tables: for each output coordinate,
        // store the corresponding input offset contribution (in_coord * in_stride).
        let mut coord_tables: [Vec<usize>; 8] = Default::default();
        for ax in 0..rank {
            let scale = out_dims[ax] as f32 / input.dims[ax] as f32;
            let max_in = (input.dims[ax] - 1) as f32;
            coord_tables[ax] = (0..out_dims[ax])
                .map(|out_coord| {
                    let orig = match self.coord_transform {
                        CoordTransform::HalfPixel => (out_coord as f32 + 0.5) / scale - 0.5,
                        CoordTransform::Asymmetric => out_coord as f32 / scale,
                        CoordTransform::AlignCorners => {
                            if out_dims[ax] <= 1 {
                                0.0
                            } else {
                                out_coord as f32 * max_in / (out_dims[ax] - 1) as f32
                            }
                        }
                    };
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

        let input_f = input.floats();
        let buf = output.as_mut_f32(numel);

        // Use incrementing coordinate array to avoid division/modulo
        let mut coord = [0usize; 8];
        let mut in_off = 0usize;
        // Initialize in_off from coord [0,0,...,0]
        for ax in 0..rank {
            in_off += coord_tables[ax][0];
        }
        for out_flat in 0..numel {
            buf[out_flat] = input_f[in_off];

            // Increment coordinate and update in_off
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

        output.set_dims(&out_dims[..rank]);
        Ok(())
    }
}
