use std::collections::HashMap;

use prost::Message;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx::ModelProto;

pub struct InferenceEngine {
    plan: Plan,
    values: HashMap<String, Tensor>,
    input_sizes: HashMap<String, Vec<usize>>,
    pub outputs: HashMap<String, Tensor>,
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

        let mut outputs = HashMap::new();
        for name in &plan.output_names {
            outputs.insert(name.clone(), Tensor::default());
        }

        Ok(Self {
            plan,
            values,
            input_sizes,
            outputs,
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<()> {
        let output_names = std::mem::take(&mut self.plan.output_names);
        let result = self.run_for(inputs, &output_names);
        self.plan.output_names = output_names;
        result
    }

    pub fn run_for(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
    ) -> Result<()> {
        let _span = tracing::trace_span!("inference").entered();

        for (k, v) in inputs {
            self.values.insert(k, v);
        }

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

        // Copy results into persistent output buffer (no allocation after warmup)
        for name in output_names {
            if let Some(src) = self.values.get(name)
                && let Some(dst) = self.outputs.get_mut(name)
            {
                dst.copy_from(src);
            }
        }

        Ok(())
    }

    pub fn input_sizes(&self) -> &HashMap<String, Vec<usize>> {
        &self.input_sizes
    }

    pub fn shape_map(&self) -> &HashMap<String, Vec<usize>> {
        &self.plan.shape_map
    }
}
