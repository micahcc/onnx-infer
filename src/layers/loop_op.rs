use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::onnx::GraphProto;

pub struct Loop {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub body: GraphProto,
}

impl Loop {
    pub fn new(inputs: Vec<String>, outputs: Vec<String>, body: GraphProto) -> Self {
        Self {
            inputs,
            outputs,
            body,
        }
    }

    pub fn execute(&self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let trip_tensor = get_tensor(values, &self.inputs[0])?;
        let trip_count = trip_tensor.i64_at(0) as usize;

        let num_carried = self.inputs.len() - 2;
        let mut carried: Vec<Tensor> = (0..num_carried)
            .map(|i| get_tensor(values, &self.inputs[i + 2]).cloned())
            .collect::<Result<Vec<_>>>()?;

        let num_scan = self.body.output.len() - 1 - num_carried;
        let mut scan_outputs: Vec<Vec<Tensor>> = vec![vec![]; num_scan];

        for i in 0..trip_count {
            let mut body_values: HashMap<String, Tensor> = HashMap::new();

            for (k, v) in values.iter() {
                body_values.insert(k.clone(), v.clone());
            }

            body_values.insert(
                self.body.input[0].name.clone(),
                Tensor::new_i64(vec![], vec![i as i64]),
            );
            body_values.insert(
                self.body.input[1].name.clone(),
                Tensor::new(vec![], vec![1.0]),
            );
            for (j, c) in carried.iter().enumerate() {
                body_values.insert(self.body.input[j + 2].name.clone(), c.clone());
            }

            for init in &self.body.initializer {
                if !init.name.is_empty() {
                    body_values.insert(init.name.clone(), Tensor::from_proto(init)?);
                }
            }

            for body_node in &self.body.node {
                crate::layers::execute_node(body_node, &mut body_values)?;
            }

            for j in 0..num_carried {
                carried[j] = body_values
                    .get(&self.body.output[j + 1].name)
                    .cloned()
                    .unwrap_or_else(|| Tensor::new(vec![], vec![]));
            }

            for j in 0..num_scan {
                let scan_val = body_values
                    .get(&self.body.output[1 + num_carried + j].name)
                    .cloned()
                    .unwrap_or_else(|| Tensor::new(vec![], vec![]));
                scan_outputs[j].push(scan_val);
            }
        }

        let mut out_idx = 0;
        for j in 0..num_carried {
            if out_idx < self.outputs.len() && !self.outputs[out_idx].is_empty() {
                values.insert(self.outputs[out_idx].clone(), carried[j].clone());
            }
            out_idx += 1;
        }
        for j in 0..num_scan {
            if out_idx < self.outputs.len() && !self.outputs[out_idx].is_empty() {
                let scans = &scan_outputs[j];
                if scans.is_empty() {
                    values.insert(self.outputs[out_idx].clone(), Tensor::new(vec![0], vec![]));
                } else {
                    let inner_dims = &scans[0].dims;
                    let mut out_dims = vec![scans.len()];
                    out_dims.extend_from_slice(inner_dims);
                    match scans[0].dtype {
                        DType::Float => {
                            let data: Vec<f32> = scans
                                .iter()
                                .flat_map(|t| (0..t.numel()).map(|i| t.f32_at(i)))
                                .collect();
                            values
                                .insert(self.outputs[out_idx].clone(), Tensor::new(out_dims, data));
                        }
                        DType::Int64 => {
                            let data: Vec<i64> = scans
                                .iter()
                                .flat_map(|t| (0..t.numel()).map(|i| t.i64_at(i)))
                                .collect();
                            values.insert(
                                self.outputs[out_idx].clone(),
                                Tensor::new_i64(out_dims, data),
                            );
                        }
                    }
                }
            }
            out_idx += 1;
        }

        Ok(())
    }
}
