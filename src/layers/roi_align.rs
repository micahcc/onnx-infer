use std::collections::HashMap;

use anyhow::Context;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

#[derive(Debug)]
pub struct RoiAlign {
    pub inputs: Vec<String>,
    pub is_avg: bool,
    pub output_height: usize,
    pub output_width: usize,
    pub sampling_ratio: usize,
    pub spatial_scale: f32,
    batch_idx_buf: Vec<usize>,
}

impl RoiAlign {
    pub fn new(
        inputs: Vec<String>,
        mode: String,
        output_height: usize,
        output_width: usize,
        sampling_ratio: usize,
        spatial_scale: f32,
    ) -> Self {
        Self {
            inputs,
            is_avg: mode != "max",
            output_height,
            output_width,
            sampling_ratio,
            spatial_scale,
            batch_idx_buf: Vec::new(),
        }
    }
}

fn bilinear_interpolate(data: &[f32], h: usize, w: usize, y: f32, x: f32) -> f32 {
    if y < -1.0 || y > h as f32 || x < -1.0 || x > w as f32 {
        return 0.0;
    }
    let y = y.max(0.0);
    let x = x.max(0.0);

    let y_low = y.floor() as usize;
    let x_low = x.floor() as usize;
    let y_high = (y_low + 1).min(h - 1);
    let x_high = (x_low + 1).min(w - 1);
    let y_low = y_low.min(h - 1);
    let x_low = x_low.min(w - 1);

    let ly = y - y_low as f32;
    let lx = x - x_low as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let v1 = data[y_low * w + x_low];
    let v2 = data[y_low * w + x_high];
    let v3 = data[y_high * w + x_low];
    let v4 = data[y_high * w + x_high];

    hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4
}

impl Layer for RoiAlign {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let x = get_tensor(values, &self.inputs[0])?;
        let rois = get_tensor(values, &self.inputs[1])?;
        let batch_indices = get_tensor(values, &self.inputs[2])?;

        let channels = x.dims[1];
        let height = x.dims[2];
        let width = x.dims[3];
        let num_rois = rois.dims[0];

        let oh = self.output_height;
        let ow = self.output_width;
        let numel = num_rois * channels * oh * ow;

        let x_data = x.floats().context("in RoiAlign layer")?;
        let rois_data = rois.floats().context("in RoiAlign layer")?;
        self.batch_idx_buf.clear();
        match batch_indices.dtype() {
            DType::Int64 => {
                for &v in batch_indices.ints().context("in RoiAlign layer")? {
                    self.batch_idx_buf.push(v as usize);
                }
            }
            DType::Float => {
                for &v in batch_indices.floats().context("in RoiAlign layer")? {
                    self.batch_idx_buf.push(v as usize);
                }
            }
            DType::String => unreachable!("strings not supported"),
        }

        let buf = output.as_mut_f32(numel);
        let is_avg = self.is_avg;

        for (n, &batch) in self.batch_idx_buf.iter().enumerate() {
            let roi_base = n * 4;
            let x1 = rois_data[roi_base] * self.spatial_scale;
            let y1 = rois_data[roi_base + 1] * self.spatial_scale;
            let x2 = rois_data[roi_base + 2] * self.spatial_scale;
            let y2 = rois_data[roi_base + 3] * self.spatial_scale;

            let roi_h = (y2 - y1).max(1.0);
            let roi_w = (x2 - x1).max(1.0);
            let bin_h = roi_h / oh as f32;
            let bin_w = roi_w / ow as f32;

            let roi_bin_grid_h = if self.sampling_ratio > 0 {
                self.sampling_ratio
            } else {
                bin_h.ceil() as usize
            };
            let roi_bin_grid_w = if self.sampling_ratio > 0 {
                self.sampling_ratio
            } else {
                bin_w.ceil() as usize
            };

            let count = (roi_bin_grid_h * roi_bin_grid_w) as f32;

            for c in 0..channels {
                let feat_base = (batch * channels + c) * height * width;
                let feat = &x_data[feat_base..feat_base + height * width];

                for ph in 0..oh {
                    for pw in 0..ow {
                        let out_idx = ((n * channels + c) * oh + ph) * ow + pw;

                        if is_avg {
                            let mut sum = 0.0f32;
                            for iy in 0..roi_bin_grid_h {
                                let y = y1
                                    + ph as f32 * bin_h
                                    + (iy as f32 + 0.5) * bin_h / roi_bin_grid_h as f32;
                                for ix in 0..roi_bin_grid_w {
                                    let x_coord = x1
                                        + pw as f32 * bin_w
                                        + (ix as f32 + 0.5) * bin_w / roi_bin_grid_w as f32;
                                    sum += bilinear_interpolate(feat, height, width, y, x_coord);
                                }
                            }
                            buf[out_idx] = sum / count;
                        } else {
                            let mut max_val = f32::NEG_INFINITY;
                            for iy in 0..roi_bin_grid_h {
                                let y = y1
                                    + ph as f32 * bin_h
                                    + (iy as f32 + 0.5) * bin_h / roi_bin_grid_h as f32;
                                for ix in 0..roi_bin_grid_w {
                                    let x_coord = x1
                                        + pw as f32 * bin_w
                                        + (ix as f32 + 0.5) * bin_w / roi_bin_grid_w as f32;
                                    max_val = max_val
                                        .max(bilinear_interpolate(feat, height, width, y, x_coord));
                                }
                            }
                            buf[out_idx] = max_val;
                        }
                    }
                }
            }
        }

        output.set_dims(&[num_rois, channels, oh, ow]);
        Ok(())
    }
}
