use std::collections::HashMap;

use prost::Message;

use crate::Result;
use crate::InferenceError;
use crate::Tensor;
use crate::layers::PlanNode;
use crate::layers::build_plan;
use crate::onnx::ModelProto;

pub struct InferenceEngine {
    plan: Vec<PlanNode>,
    initializers: HashMap<String, Tensor>,
    output_names: Vec<String>,
    input_sizes: HashMap<String, Vec<usize>>,
    shape_map: HashMap<String, Vec<usize>>,
    tensor_pool: HashMap<String, Tensor>,
}

impl InferenceEngine {
    pub fn new(model_bytes: &[u8], input_sizes: HashMap<String, Vec<usize>>) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).map_err(InferenceError::ParseError)?;
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| InferenceError::InvalidModel("Model has no graph".into()))?;

        let (plan, initializers, output_names, shape_map, tensor_pool) =
            build_plan(graph, &input_sizes)?;

        Ok(Self {
            plan,
            initializers,
            output_names,
            input_sizes,
            shape_map,
            tensor_pool,
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let output_names = self.output_names.clone();
        self.run_with_outputs(inputs, &output_names)
    }

    pub fn run_with_outputs(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let _span = tracing::trace_span!("inference").entered();

        let mut values: HashMap<String, Tensor> = inputs;

        // Load initializers
        for (k, v) in &self.initializers {
            values.insert(k.clone(), v.clone());
        }

        // Insert pre-allocated tensors for outputs that don't already exist
        for (k, v) in &self.tensor_pool {
            if !values.contains_key(k) {
                values.insert(k.clone(), v.clone());
            }
        }

        // Execute plan
        for node in &mut self.plan {
            match node {
                PlanNode::Single { output, layer } => {
                    if output.is_empty() {
                        continue;
                    }
                    let _span = tracing::trace_span!("op").entered();
                    let mut out = values.remove(output.as_str()).unwrap_or_default();
                    let result = layer.execute(&values, &mut out);
                    values.insert(output.clone(), out);
                    result?;
                }
                PlanNode::Loop(loop_layer) => {
                    loop_layer.execute(&mut values)?;
                }
            }
        }

        // Collect requested outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            if let Some(tensor) = values.get(name) {
                outputs.insert(name.clone(), tensor.clone());
            }
        }

        Ok(outputs)
    }

    pub fn input_sizes(&self) -> &HashMap<String, Vec<usize>> {
        &self.input_sizes
    }

    pub fn shape_map(&self) -> &HashMap<String, Vec<usize>> {
        &self.shape_map
    }
}
