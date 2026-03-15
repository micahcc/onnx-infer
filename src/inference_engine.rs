use std::collections::HashMap;

use prost::Message;

use crate::Result;
use crate::InferenceError;
use crate::Tensor;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx::ModelProto;

pub struct InferenceEngine {
    plan: Plan,
    values: HashMap<String, Tensor>,
    input_sizes: HashMap<String, Vec<usize>>,
}

impl InferenceEngine {
    pub fn new(model_bytes: &[u8], input_sizes: HashMap<String, Vec<usize>>) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).map_err(InferenceError::ParseError)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| InferenceError::InvalidModel("Model has no graph".into()))?;

        let mut plan = Plan::build(graph, &input_sizes)?;

        let mut values = HashMap::new();
        for (k, v) in std::mem::take(&mut plan.initializers) {
            values.insert(k, v);
        }
        for (k, v) in std::mem::take(&mut plan.tensor_pool) {
            values.insert(k, v);
        }

        Ok(Self {
            plan,
            values,
            input_sizes,
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Take ownership temporarily to avoid borrow conflict
        let output_names = std::mem::take(&mut self.plan.output_names);
        let result = self.run_with_outputs(inputs, &output_names);
        self.plan.output_names = output_names;
        result
    }

    pub fn run_with_outputs(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let _span = tracing::trace_span!("inference").entered();

        for (k, v) in inputs {
            self.values.insert(k, v);
        }

        // Execute plan
        for node in &mut self.plan.nodes {
            match node {
                PlanNode::Single { output, layer } => {
                    if output.is_empty() {
                        continue;
                    }
                    let _span = tracing::trace_span!("op").entered();
                    let (key, mut out) = self
                        .values
                        .remove_entry(output.as_str())
                        .unwrap_or_else(|| (output.clone(), Tensor::default()));
                    let result = layer.execute(&self.values, &mut out);
                    self.values.insert(key, out);
                    result?;
                }
                PlanNode::Loop(loop_layer) => {
                    loop_layer.execute(&mut self.values)?;
                }
            }
        }

        // Collect requested outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            if let Some(tensor) = self.values.get(name) {
                outputs.insert(name.clone(), tensor.clone());
            }
        }

        Ok(outputs)
    }

    pub fn input_sizes(&self) -> &HashMap<String, Vec<usize>> {
        &self.input_sizes
    }

    pub fn shape_map(&self) -> &HashMap<String, Vec<usize>> {
        &self.plan.shape_map
    }
}
