use std::collections::HashMap;

use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Layer;

pub struct Nms {
    pub inputs: Vec<String>,
    selected: Vec<[i64; 3]>,
    candidates: Vec<(usize, f32)>,
    kept: Vec<usize>,
}

impl Nms {
    pub fn new(inputs: Vec<String>) -> Self {
        Self {
            inputs,
            selected: Vec::new(),
            candidates: Vec::new(),
            kept: Vec::new(),
        }
    }
}

impl Layer for Nms {
    fn execute(&mut self, values: &HashMap<String, Tensor>, output: &mut Tensor) -> Result<()> {
        let boxes = get_tensor(values, &self.inputs[0])?;
        let scores = get_tensor(values, &self.inputs[1])?;
        let max_output = if self.inputs.len() > 2 && !self.inputs[2].is_empty() {
            get_tensor(values, &self.inputs[2])?.i64_at(0) as usize
        } else {
            0
        };
        let iou_threshold = if self.inputs.len() > 3 && !self.inputs[3].is_empty() {
            get_tensor(values, &self.inputs[3])?.floats()[0]
        } else {
            0.0
        };
        let score_threshold = if self.inputs.len() > 4 && !self.inputs[4].is_empty() {
            get_tensor(values, &self.inputs[4])?.floats()[0]
        } else {
            f32::NEG_INFINITY
        };

        let batches = boxes.dims[0];
        let num_boxes = boxes.dims[1];
        let num_classes = scores.dims[1];

        let boxes_f = boxes.floats();
        let scores_f = scores.floats();
        self.selected.clear();

        for batch in 0..batches {
            for class in 0..num_classes {
                self.candidates.clear();
                for b in 0..num_boxes {
                    let score = scores_f[(batch * num_classes + class) * num_boxes + b];
                    if score > score_threshold {
                        self.candidates.push((b, score));
                    }
                }
                self.candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                self.kept.clear();
                for &(box_idx, _) in &self.candidates {
                    if max_output > 0 && self.kept.len() >= max_output {
                        break;
                    }
                    let base_i = (batch * num_boxes + box_idx) * 4;
                    let bi = &boxes_f[base_i..base_i + 4];

                    let mut suppress = false;
                    for &k in &self.kept {
                        let base_k = (batch * num_boxes + k) * 4;
                        let bk = &boxes_f[base_k..base_k + 4];
                        if iou(bi, bk) > iou_threshold {
                            suppress = true;
                            break;
                        }
                    }
                    if !suppress {
                        self.kept.push(box_idx);
                        self.selected.push([batch as i64, class as i64, box_idx as i64]);
                    }
                }
            }
        }

        let num_selected = self.selected.len();
        let buf = output.as_mut_i64(num_selected * 3);
        for (i, s) in self.selected.iter().enumerate() {
            buf[i * 3] = s[0];
            buf[i * 3 + 1] = s[1];
            buf[i * 3 + 2] = s[2];
        }
        output.set_dims(&[num_selected, 3]);
        Ok(())
    }
}

fn iou(a: &[f32], b: &[f32]) -> f32 {
    let y1 = a[0].max(b[0]);
    let x1 = a[1].max(b[1]);
    let y2 = a[2].min(b[2]);
    let x2 = a[3].min(b[3]);
    let inter = (y2 - y1).max(0.0) * (x2 - x1).max(0.0);
    let area_a = (a[2] - a[0]).abs() * (a[3] - a[1]).abs();
    let area_b = (b[2] - b[0]).abs() * (b[3] - b[1]).abs();
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}
