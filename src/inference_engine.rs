use std::collections::HashMap;

use anyhow::Context;
use prost::Message;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx::ModelProto;
use crate::onnx_ir;

pub struct InferenceEngine {
    graph: onnx_ir::Graph,
    plan: Option<Plan>,
    values: HashMap<String, Tensor>,
    input_names: Vec<String>,
    input_sizes: HashMap<String, Dims>,
    pub outputs: HashMap<String, Tensor>,
    #[cfg(feature = "xnnpack")]
    use_xnnpack: bool,
}

impl InferenceEngine {
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
        Self::with_batch_size(model_bytes, 1)
    }

    /// Create an engine with a fixed batch size.
    ///
    /// Models with dynamic batch dimensions (common in ONNX exports) use
    /// symbolic dimension names like `"N"` or `"batch"`. This method resolves
    /// all such dimensions to the given `batch_size`, enabling full shape
    /// inference at build time, enabling full shape inference for spatial
    /// ops like Conv and MaxPool.
    ///
    /// ```no_run
    /// # use onnx_infer::InferenceEngine;
    /// let model_bytes = std::fs::read("model.onnx").unwrap();
    /// let mut engine = InferenceEngine::with_batch_size(&model_bytes, 1).unwrap();
    /// ```
    pub fn with_batch_size(model_bytes: &[u8], batch_size: usize) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).context("decoding model proto")?;
        let opset_version = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty())
            .map(|o| o.version)
            .max()
            .unwrap_or(0);
        let graph_proto = model.graph.as_ref().context("model has no graph")?;
        let graph = onnx_ir::convert_graph_with_opset(graph_proto, opset_version)?;

        let initializer_names: std::collections::HashSet<&str> =
            graph.initializers.keys().map(|k| k.as_str()).collect();

        let mut input_sizes = HashMap::new();
        for input in &graph.inputs {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = &input.shape {
                let mut shape = shape.clone();
                if !shape.is_empty() && shape[0] == 0 {
                    shape[0] = batch_size;
                }
                if shape.iter().all(|&d| d > 0) {
                    input_sizes.insert(input.name.clone(), shape);
                }
            }
        }

        Self::build_from_graph(graph, input_sizes)
    }

    pub fn with_input_sizes(
        model_bytes: &[u8],
        input_sizes: HashMap<String, Dims>,
    ) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).context("decoding model proto")?;
        let opset_version = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty())
            .map(|o| o.version)
            .max()
            .unwrap_or(0);
        let graph_proto = model.graph.as_ref().context("model has no graph")?;
        let graph = onnx_ir::convert_graph_with_opset(graph_proto, opset_version)?;

        Self::build_from_graph(graph, input_sizes)
    }

    /// Create an engine with CPU-safe graph optimizations applied (e.g. BN fold into Conv).
    pub fn with_graph_opt(model_bytes: &[u8]) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).context("decoding model proto")?;
        let opset_version = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty())
            .map(|o| o.version)
            .max()
            .unwrap_or(0);
        let graph_proto = model.graph.as_ref().context("model has no graph")?;
        let mut graph = onnx_ir::convert_graph_with_opset(graph_proto, opset_version)?;

        crate::graph_opt::optimize_cpu(&mut graph);

        let initializer_names: std::collections::HashSet<&str> =
            graph.initializers.keys().map(|k| k.as_str()).collect();

        let mut input_sizes = HashMap::new();
        for input in &graph.inputs {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = &input.shape {
                let mut shape = shape.clone();
                if !shape.is_empty() && shape[0] == 0 {
                    shape[0] = 1;
                }
                if shape.iter().all(|&d| d > 0) {
                    input_sizes.insert(input.name.clone(), shape);
                }
            }
        }

        Self::build_from_graph(graph, input_sizes)
    }

    /// Create an engine with XNNPACK acceleration.
    ///
    /// Applies full graph optimizations including NHWC layout transposes,
    /// then compiles eligible op sequences into XNNPACK subgraphs.
    #[cfg(feature = "xnnpack")]
    pub fn with_xnnpack(model_bytes: &[u8]) -> Result<Self> {
        let model = ModelProto::decode(model_bytes).context("decoding model proto")?;
        let opset_version = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty())
            .map(|o| o.version)
            .max()
            .unwrap_or(0);
        let graph_proto = model.graph.as_ref().context("model has no graph")?;
        let mut graph = onnx_ir::convert_graph_with_opset(graph_proto, opset_version)?;

        crate::graph_opt::optimize(&mut graph);

        let initializer_names: std::collections::HashSet<&str> =
            graph.initializers.keys().map(|k| k.as_str()).collect();

        let mut input_sizes = HashMap::new();
        for input in &graph.inputs {
            if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                continue;
            }
            if let Some(shape) = &input.shape {
                let mut shape = shape.clone();
                if !shape.is_empty() && shape[0] == 0 {
                    shape[0] = 1;
                }
                if shape.iter().all(|&d| d > 0) {
                    input_sizes.insert(input.name.clone(), shape);
                }
            }
        }

        Self::build_from_graph_xnnpack(graph, input_sizes)
    }

    /// Dump the current (possibly optimized) IR graph as human-readable text.
    pub fn dump_graph(&self) -> String {
        crate::graph_opt::dump(&self.graph)
    }

    /// Parse a model and return the pre-optimization and post-optimization graph dumps.
    /// The optimization includes NHWC layout transposes (for XNNPACK).
    pub fn dump_graph_opt(model_bytes: &[u8]) -> Result<(String, String)> {
        let model = ModelProto::decode(model_bytes).context("decoding model proto")?;
        let opset_version = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty())
            .map(|o| o.version)
            .max()
            .unwrap_or(0);
        let graph_proto = model.graph.as_ref().context("model has no graph")?;
        let mut graph = onnx_ir::convert_graph_with_opset(graph_proto, opset_version)?;

        let before = crate::graph_opt::dump(&graph);
        crate::graph_opt::optimize(&mut graph);
        let after = crate::graph_opt::dump(&graph);

        Ok((before, after))
    }

    /// Parse a model and return the pre-optimization and post-CPU-optimization graph dumps.
    /// CPU optimization includes BN folding but no layout transposes.
    pub fn dump_graph_opt_cpu(model_bytes: &[u8]) -> Result<(String, String)> {
        let model = ModelProto::decode(model_bytes).context("decoding model proto")?;
        let opset_version = model
            .opset_import
            .iter()
            .filter(|o| o.domain.is_empty())
            .map(|o| o.version)
            .max()
            .unwrap_or(0);
        let graph_proto = model.graph.as_ref().context("model has no graph")?;
        let mut graph = onnx_ir::convert_graph_with_opset(graph_proto, opset_version)?;

        let before = crate::graph_opt::dump(&graph);
        crate::graph_opt::optimize_cpu(&mut graph);
        let after = crate::graph_opt::dump(&graph);

        Ok((before, after))
    }

    #[cfg(feature = "xnnpack")]
    fn build_from_graph_xnnpack(
        graph: onnx_ir::Graph,
        input_sizes: HashMap<String, Dims>,
    ) -> Result<Self> {
        let initializer_names: std::collections::HashSet<&str> =
            graph.initializers.keys().map(|k| k.as_str()).collect();
        let input_names: Vec<String> = graph
            .inputs
            .iter()
            .filter(|i| !i.name.is_empty() && !initializer_names.contains(i.name.as_str()))
            .map(|i| i.name.clone())
            .collect();

        let output_names: Vec<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();
        let mut outputs = HashMap::new();
        for name in &output_names {
            outputs.insert(name.clone(), Tensor::default());
        }

        let all_shapes_known = input_names.iter().all(|n| input_sizes.contains_key(n));
        let plan = if all_shapes_known {
            Some(Plan::build_with_xnnpack(&graph, &input_sizes)?)
        } else {
            None
        };

        let mut values = HashMap::new();
        if let Some(ref plan) = plan {
            Self::load_plan_values(&mut values, plan);
        }

        Ok(Self {
            graph,
            plan,
            values,
            input_names,
            input_sizes,
            outputs,
            #[cfg(feature = "xnnpack")]
            use_xnnpack: true,
        })
    }

    fn build_from_graph(graph: onnx_ir::Graph, input_sizes: HashMap<String, Dims>) -> Result<Self> {
        let initializer_names: std::collections::HashSet<&str> =
            graph.initializers.keys().map(|k| k.as_str()).collect();
        let input_names: Vec<String> = graph
            .inputs
            .iter()
            .filter(|i| !i.name.is_empty() && !initializer_names.contains(i.name.as_str()))
            .map(|i| i.name.clone())
            .collect();

        let output_names: Vec<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();
        let mut outputs = HashMap::new();
        for name in &output_names {
            outputs.insert(name.clone(), Tensor::default());
        }

        // Build plan eagerly if all input shapes are known
        let all_shapes_known = input_names.iter().all(|n| input_sizes.contains_key(n));
        let plan = if all_shapes_known {
            Some(Plan::build(&graph, &input_sizes)?)
        } else {
            None
        };

        let mut values = HashMap::new();
        if let Some(ref plan) = plan {
            Self::load_plan_values(&mut values, plan);
        }

        Ok(Self {
            graph,
            plan,
            values,
            input_names,
            input_sizes,
            outputs,
            #[cfg(feature = "xnnpack")]
            use_xnnpack: false,
        })
    }

    fn load_plan_values(values: &mut HashMap<String, Tensor>, plan: &Plan) {
        for (k, v) in &plan.initializers {
            values.insert(k.clone(), v.clone());
        }
        for (k, v) in &plan.tensor_pool {
            values.insert(k.clone(), v.clone());
        }
    }

    fn ensure_plan(&mut self, inputs: &HashMap<String, Tensor>) -> Result<()> {
        // Check if we need to rebuild: no plan, or input shapes changed
        let needs_rebuild = match &self.plan {
            None => true,
            Some(_) => {
                // Check if any input shape differs from what we built with
                inputs.iter().any(|(name, tensor)| {
                    self.input_sizes
                        .get(name)
                        .is_none_or(|s| s.as_slice() != tensor.dims.as_slice())
                })
            }
        };

        if !needs_rebuild {
            return Ok(());
        }

        // Derive input_sizes from actual input tensors
        let mut input_sizes = self.input_sizes.clone();
        for (name, tensor) in inputs {
            input_sizes.insert(name.clone(), tensor.dims.clone());
        }

        // Build plan with actual input values for aggressive constant folding
        #[cfg(feature = "xnnpack")]
        let plan = if self.use_xnnpack {
            Plan::build_with_xnnpack(&self.graph, &input_sizes)?
        } else {
            Plan::build_full(&self.graph, &input_sizes, &HashMap::new(), inputs)?
        };
        #[cfg(not(feature = "xnnpack"))]
        let plan = Plan::build_full(&self.graph, &input_sizes, &HashMap::new(), inputs)?;

        // Reset values and reload from new plan
        self.values.clear();
        Self::load_plan_values(&mut self.values, &plan);

        // Update cached input sizes
        for (name, tensor) in inputs {
            self.input_sizes.insert(name.clone(), tensor.dims.clone());
        }

        self.plan = Some(plan);
        Ok(())
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<()> {
        self.ensure_plan(&inputs)?;
        let plan = self.plan.as_mut().unwrap();
        let output_names = std::mem::take(&mut plan.output_names);
        let mut outputs = std::mem::take(&mut self.outputs);
        let result = self.run_inner(&inputs, &output_names, &mut outputs);
        self.plan.as_mut().unwrap().output_names = output_names;
        self.outputs = outputs;
        result?;
        Ok(())
    }

    fn run_inner(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        output_names: &[String],
        outputs: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("inference").entered();

        for (k, v) in inputs {
            self.values.insert(k.clone(), v.clone());
        }

        let plan = self.plan.as_mut().unwrap();
        for node in &mut plan.nodes {
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
                #[cfg(feature = "xnnpack")]
                PlanNode::XnnpackSubgraph(sg) => {
                    sg.execute(&mut self.values)?;
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

    pub fn run_for(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
        outputs: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        self.ensure_plan(&inputs)?;
        for (k, v) in &inputs {
            self.values.insert(k.clone(), v.clone());
        }

        let plan = self.plan.as_mut().unwrap();
        for node in &mut plan.nodes {
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
                #[cfg(feature = "xnnpack")]
                PlanNode::XnnpackSubgraph(sg) => {
                    sg.execute(&mut self.values)?;
                }
            }
        }

        for name in output_names {
            if let Some(src) = self.values.get(name) {
                let dst = outputs.entry(name.clone()).or_default();
                dst.copy_from(src);
            }
        }

        Ok(())
    }

    pub fn input_sizes(&self) -> HashMap<String, Dims> {
        let shape_map = self.plan.as_ref().map(|p| &p.shape_map);
        self.input_names
            .iter()
            .filter_map(|name| {
                shape_map
                    .and_then(|sm| sm.get(name))
                    .or_else(|| self.input_sizes.get(name))
                    .map(|dims| (name.clone(), dims.clone()))
            })
            .collect()
    }

    pub fn value(&self, name: &str) -> Option<&Tensor> {
        self.values.get(name)
    }

    pub fn shape_map(&self) -> HashMap<String, Dims> {
        self.plan
            .as_ref()
            .map(|p| p.shape_map.clone())
            .unwrap_or_default()
    }
}
