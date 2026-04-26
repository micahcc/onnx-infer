use std::collections::HashMap;

use anyhow::Context;
use prost::Message;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::TensorData;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx::ModelProto;
use crate::onnx_ir;

/// Options for creating an [`InferenceEngine`].
///
/// ```no_run
/// # use onnx_infer::{InferenceEngine, InferenceOptions};
/// let model_bytes = std::fs::read("model.onnx").unwrap();
/// let mut engine = InferenceEngine::new(&model_bytes, InferenceOptions::default()).unwrap();
/// ```
///
/// Enable XNNPACK acceleration:
/// ```no_run
/// # use onnx_infer::{InferenceEngine, InferenceOptions};
/// # let model_bytes = std::fs::read("model.onnx").unwrap();
/// let mut engine = InferenceEngine::new(
///     &model_bytes,
///     InferenceOptions { xnnpack: true, ..Default::default() },
/// ).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct InferenceOptions {
    /// Batch size for models with dynamic batch dimensions. Default: 1.
    pub batch_size: usize,

    /// Explicit input sizes. When set, overrides shape inference from the model.
    pub input_sizes: Option<HashMap<String, Dims>>,

    /// Enable XNNPACK acceleration (requires the `xnnpack` feature). Default: false.
    pub xnnpack: bool,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            input_sizes: None,
            xnnpack: false,
        }
    }
}

pub struct InferenceEngine {
    graph: onnx_ir::Graph,
    plan: Option<Plan>,
    values: HashMap<String, Tensor>,
    input_names: Vec<String>,
    input_sizes: HashMap<String, Dims>,
    pub outputs: HashMap<String, Tensor>,
    use_xnnpack: bool,
}

impl InferenceEngine {
    /// Create a new inference engine from model bytes and options.
    pub fn new(model_bytes: &[u8], opts: InferenceOptions) -> Result<Self> {
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

        let input_sizes = if let Some(sizes) = opts.input_sizes {
            sizes
        } else {
            let initializer_names: std::collections::HashSet<&str> =
                graph.initializers.keys().map(|k| k.as_str()).collect();
            let mut sizes = HashMap::new();
            for input in &graph.inputs {
                if input.name.is_empty() || initializer_names.contains(input.name.as_str()) {
                    continue;
                }
                if let Some(shape) = &input.shape {
                    let mut shape = shape.clone();
                    if !shape.is_empty() && shape[0] == 0 {
                        shape[0] = opts.batch_size;
                    }
                    if shape.iter().all(|&d| d > 0) {
                        sizes.insert(input.name.clone(), shape);
                    }
                }
            }
            sizes
        };

        #[cfg(not(feature = "xnnpack"))]
        if opts.xnnpack {
            tracing::warn!(
                "xnnpack requested but onnx-infer was compiled without the `xnnpack` feature — falling back to CPU"
            );
        }

        Self::build_from_graph(graph, input_sizes, opts.xnnpack)
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

    fn build_from_graph(
        graph: onnx_ir::Graph,
        input_sizes: HashMap<String, Dims>,
        use_xnnpack: bool,
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
            #[cfg(feature = "xnnpack")]
            if use_xnnpack {
                Some(Plan::build_with_xnnpack(
                    &graph,
                    &input_sizes,
                    &HashMap::new(),
                )?)
            } else {
                Some(Plan::build(&graph, &input_sizes)?)
            }
            #[cfg(not(feature = "xnnpack"))]
            {
                let _ = use_xnnpack;
                Some(Plan::build(&graph, &input_sizes)?)
            }
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
            use_xnnpack,
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
        // Check if we need to rebuild: no plan, or input shapes changed.
        // Check both explicit inputs and values already in self.values
        // (e.g. written via input_floats_mut).
        let needs_rebuild = match &self.plan {
            None => true,
            Some(_) => inputs
                .iter()
                .chain(
                    self.input_names
                        .iter()
                        .filter_map(|n| self.values.get(n).map(|t| (n, t))),
                )
                .any(|(name, tensor)| {
                    self.input_sizes
                        .get(name)
                        .is_none_or(|s| s.as_slice() != tensor.dims.as_slice())
                }),
        };

        if !needs_rebuild {
            return Ok(());
        }

        // Derive input_sizes from actual input tensors
        let mut input_sizes = self.input_sizes.clone();
        for name in &self.input_names {
            if let Some(tensor) = self.values.get(name) {
                input_sizes.insert(name.clone(), tensor.dims.clone());
            }
        }
        for (name, tensor) in inputs {
            input_sizes.insert(name.clone(), tensor.dims.clone());
        }

        // Build plan with actual input values for aggressive constant folding
        #[cfg(feature = "xnnpack")]
        let plan = if self.use_xnnpack {
            Plan::build_with_xnnpack(&self.graph, &input_sizes, inputs)?
        } else {
            Plan::build_full(&self.graph, &input_sizes, &HashMap::new(), inputs)?
        };
        #[cfg(not(feature = "xnnpack"))]
        let plan = {
            let _ = self.use_xnnpack;
            Plan::build_full(&self.graph, &input_sizes, &HashMap::new(), inputs)?
        };

        // Preserve input tensors written via input_floats_mut before clearing
        let saved_inputs: Vec<(String, Tensor)> = self
            .input_names
            .iter()
            .filter_map(|n| self.values.remove_entry(n))
            .collect();

        // Reset values and reload from new plan
        self.values.clear();
        Self::load_plan_values(&mut self.values, &plan);

        // Restore saved input tensors
        for (name, tensor) in saved_inputs {
            self.values.insert(name, tensor);
        }

        // Update cached input sizes
        for (name, tensor) in inputs {
            self.input_sizes.insert(name.clone(), tensor.dims.clone());
        }
        // Also update from restored inputs
        for name in &self.input_names {
            if let Some(tensor) = self.values.get(name) {
                self.input_sizes.insert(name.clone(), tensor.dims.clone());
            }
        }

        self.plan = Some(plan);
        Ok(())
    }

    /// Run inference using inputs already written into the engine via
    /// [`input_floats_mut`](Self::input_floats_mut). No input tensors are
    /// allocated or copied — the engine executes directly on its internal
    /// buffers.
    /// Run inference using inputs already written into the engine via
    /// [`input_floats_mut`](Self::input_floats_mut). No input tensors are
    /// allocated or copied — the engine executes directly on its internal
    /// buffers.
    pub fn run_planned(&mut self) -> Result<&HashMap<String, Tensor>> {
        self.ensure_plan(&HashMap::new())?;
        let plan = self.plan.as_mut().unwrap();
        let output_names = std::mem::take(&mut plan.output_names);
        let mut outputs = std::mem::take(&mut self.outputs);
        let result = self.run_inner(&HashMap::new(), &output_names, &mut outputs);
        self.plan.as_mut().unwrap().output_names = output_names;
        self.outputs = outputs;
        result?;
        Ok(&self.outputs)
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<&HashMap<String, Tensor>> {
        self.ensure_plan(&inputs)?;
        let plan = self.plan.as_mut().unwrap();
        let output_names = std::mem::take(&mut plan.output_names);
        let mut outputs = std::mem::take(&mut self.outputs);
        let result = self.run_inner(&inputs, &output_names, &mut outputs);
        self.plan.as_mut().unwrap().output_names = output_names;
        self.outputs = outputs;
        result?;
        Ok(&self.outputs)
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
        for (k, v) in inputs {
            self.values.insert(k, v);
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
                    .map(|sl| &sl.dims)
                    .or_else(|| self.input_sizes.get(name))
                    .map(|dims| (name.clone(), dims.clone()))
            })
            .collect()
    }

    pub fn value(&self, name: &str) -> Option<&Tensor> {
        self.values.get(name)
    }

    /// Returns a mutable slice to the f32 data of an input tensor, reusing
    /// the existing allocation when possible. The tensor is resized to match
    /// `dims`; no allocation occurs if the capacity is already sufficient.
    pub fn input_floats_mut(&mut self, name: &str, dims: Dims) -> Result<&mut [f32]> {
        let numel: usize = dims.iter().product();
        let tensor = self
            .values
            .entry(name.to_string())
            .or_insert_with(|| Tensor::new(dims.clone(), vec![0.0; numel]));
        tensor.dims = dims;
        match &mut tensor.data {
            TensorData::F32(buf) => {
                buf.resize(numel, 0.0);
                Ok(buf.as_mut_slice())
            }
            _ => anyhow::bail!("expected f32 tensor for input '{name}'"),
        }
    }

    pub fn shape_map(&self) -> HashMap<String, Dims> {
        self.plan
            .as_ref()
            .map(|p| {
                p.shape_map
                    .iter()
                    .map(|(k, v)| (k.clone(), v.dims.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }
}
