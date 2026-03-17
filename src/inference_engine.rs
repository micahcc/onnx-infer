use std::collections::HashMap;

use prost::Message;

use crate::Dims;
use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx::ModelProto;

pub struct InferenceEngine {
    plan: Plan,
    values: HashMap<String, Tensor>,
    input_names: Vec<String>,
    pub outputs: HashMap<String, Tensor>,
}

impl InferenceEngine {
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
        Self::with_input_sizes(model_bytes, HashMap::new())
    }

    pub fn with_input_sizes(
        model_bytes: &[u8],
        input_sizes: HashMap<String, Dims>,
    ) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).map_err(InferenceError::ParseError)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| InferenceError::InvalidModel("Model has no graph".into()))?;

        let mut plan = Plan::build(graph, &input_sizes)?;

        let initializer_names: std::collections::HashSet<&str> =
            graph.initializer.iter().map(|i| i.name.as_str()).collect();
        let input_names: Vec<String> = graph
            .input
            .iter()
            .filter(|i| !i.name.is_empty() && !initializer_names.contains(i.name.as_str()))
            .map(|i| i.name.clone())
            .collect();

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
            input_names,
            outputs,
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<()> {
        let output_names = std::mem::take(&mut self.plan.output_names);
        let mut outputs = std::mem::take(&mut self.outputs);
        let result = self.run_for(inputs, &output_names, &mut outputs);
        self.plan.output_names = output_names;
        self.outputs = outputs;
        result
    }

    pub fn run_for(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
        outputs: &mut HashMap<String, Tensor>,
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
                    let _span = tracing::trace_span!("op", output = %output).entered();
                    let (key, mut out) = self
                        .values
                        .remove_entry(output.as_str())
                        .unwrap_or_else(|| (output.clone(), Tensor::default()));
                    layer.execute(&self.values, &mut out)?;
                    self.values.insert(key, out);
                }
                PlanNode::Loop(loop_layer) => {
                    loop_layer.execute(&mut self.values)?;
                }
                PlanNode::Split(split_layer) => {
                    split_layer.execute(&mut self.values)?;
                }
                PlanNode::If(if_layer) => {
                    if_layer.execute(&mut self.values)?;
                }
                PlanNode::TopK(topk_layer) => {
                    topk_layer.execute(&mut self.values)?;
                }
                PlanNode::Scan(scan_layer) => {
                    scan_layer.execute(&mut self.values)?;
                }
            }
        }

        // Copy results into caller-provided output buffer
        for name in output_names {
            if let Some(src) = self.values.get(name) {
                let dst = outputs.entry(name.clone()).or_default();
                dst.copy_from(src);
            }
        }

        Ok(())
    }

    pub fn input_sizes(&self) -> HashMap<String, Dims> {
        self.input_names
            .iter()
            .filter_map(|name| {
                self.plan
                    .shape_map
                    .get(name)
                    .map(|dims| (name.clone(), dims.clone()))
            })
            .collect()
    }

    pub fn shape_map(&self) -> &HashMap<String, Dims> {
        &self.plan.shape_map
    }
}
