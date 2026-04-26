use std::collections::HashMap;

use anyhow::Context;
use prost::Message;

use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::TensorData;
use crate::Values;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx::ModelProto;
use crate::onnx_ir;

fn plan_matches_inputs(
    input_names: &[String],
    inputs: &HashMap<String, Tensor>,
    plan: &Plan,
) -> bool {
    input_names
        .iter()
        .all(|name| match (inputs.get(name), plan.shape_map.get(name)) {
            (Some(tensor), Some(sl)) => tensor.dims.as_slice() == sl.dims.as_slice(),
            (None, None) => true,
            _ => false,
        })
}

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
    /// Enable XNNPACK acceleration (requires the `xnnpack` feature). Default: false.
    pub xnnpack: bool,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self { xnnpack: false }
    }
}

pub struct InferenceEngine {
    graph: onnx_ir::Graph,
    plan_cache: Vec<Plan>,
    current_plan: Option<usize>,
    input_names: Vec<String>,
    use_xnnpack: bool,

    /// Graph weights + folded constants from plan building. Shared across plans.
    initializers: HashMap<String, Tensor>,

    /// Pre-allocated intermediate buffers + execution results. Swapped per plan.
    intermediates: HashMap<String, Tensor>,

    /// User-provided input tensors.
    pub inputs: HashMap<String, Tensor>,

    pub outputs: HashMap<String, Tensor>,
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

        #[cfg(not(feature = "xnnpack"))]
        if opts.xnnpack {
            tracing::warn!(
                "xnnpack requested but onnx-infer was compiled without the `xnnpack` feature — falling back to CPU"
            );
        }

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

        let initializers = std::mem::take(&mut graph.initializers);

        Ok(Self {
            graph,
            plan_cache: Vec::new(),
            current_plan: None,
            input_names,
            initializers,
            intermediates: HashMap::new(),
            inputs: HashMap::new(),
            outputs,
            use_xnnpack: opts.xnnpack,
        })
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

    /// Find or build a plan matching the current input shapes.
    fn ensure_plan(&mut self) -> Result<()> {
        if let Some(idx) = self.current_plan {
            if plan_matches_inputs(&self.input_names, &self.inputs, &self.plan_cache[idx]) {
                return Ok(());
            }
        }

        for (idx, plan) in self.plan_cache.iter().enumerate() {
            if plan_matches_inputs(&self.input_names, &self.inputs, plan) {
                self.switch_to_plan(idx);
                return Ok(());
            }
        }

        let mut input_sizes = HashMap::new();
        for name in &self.input_names {
            if let Some(tensor) = self.inputs.get(name) {
                input_sizes.insert(name.clone(), tensor.dims.clone());
            }
        }

        #[cfg(feature = "xnnpack")]
        let plan = if self.use_xnnpack {
            Plan::build_with_xnnpack(
                &self.graph,
                &input_sizes,
                &self.inputs,
                &mut self.initializers,
            )?
        } else {
            Plan::build_full(
                &self.graph,
                &input_sizes,
                &HashMap::new(),
                &self.inputs,
                &mut self.initializers,
            )?
        };
        #[cfg(not(feature = "xnnpack"))]
        let plan = {
            let _ = self.use_xnnpack;
            Plan::build_full(
                &self.graph,
                &input_sizes,
                &HashMap::new(),
                &self.inputs,
                &mut self.initializers,
            )?
        };

        let idx = self.plan_cache.len();
        self.plan_cache.push(plan);
        self.switch_to_plan(idx);
        Ok(())
    }

    fn switch_to_plan(&mut self, idx: usize) {
        self.current_plan = Some(idx);
    }

    fn current_plan_mut(&mut self) -> &mut Plan {
        &mut self.plan_cache[self.current_plan.unwrap()]
    }

    /// Run inference using inputs already written via [`input_floats_mut`](Self::input_floats_mut).
    pub fn run_planned(&mut self) -> Result<&HashMap<String, Tensor>> {
        self.ensure_plan()?;
        let output_names = std::mem::take(&mut self.current_plan_mut().output_names);
        let mut outputs = std::mem::take(&mut self.outputs);
        let result = self.execute_plan(&output_names, &mut outputs);
        self.current_plan_mut().output_names = output_names;
        self.outputs = outputs;
        result?;
        Ok(&self.outputs)
    }

    pub fn run(&mut self, inputs: HashMap<String, Tensor>) -> Result<&HashMap<String, Tensor>> {
        for (k, v) in inputs {
            self.inputs.insert(k, v);
        }
        self.run_planned()
    }

    pub fn run_for(
        &mut self,
        inputs: HashMap<String, Tensor>,
        output_names: &[String],
        outputs: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        for (k, v) in inputs {
            self.inputs.insert(k, v);
        }
        self.ensure_plan()?;
        self.execute_plan(output_names, outputs)
    }

    fn execute_plan(
        &mut self,
        output_names: &[String],
        outputs: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("inference").entered();

        let plan = &mut self.plan_cache[self.current_plan.unwrap()];
        for node in &mut plan.nodes {
            match node {
                PlanNode::Single { output, layer } => {
                    if output.is_empty() {
                        continue;
                    }
                    let _span = tracing::trace_span!("op", output = %output).entered();
                    let (key, mut out) = self
                        .intermediates
                        .remove_entry(output.as_str())
                        .unwrap_or_else(|| (output.clone(), Tensor::default()));
                    let values = Values {
                        intermediates: &self.intermediates,
                        inputs: &self.inputs,
                        initializers: &self.initializers,
                    };
                    layer.execute(&values, &mut out)?;
                    self.intermediates.insert(key, out);
                }
                PlanNode::Loop(loop_layer) => {
                    loop_layer.execute(
                        &mut self.intermediates,
                        &self.inputs,
                        &self.initializers,
                    )?;
                }
                PlanNode::Split(split_layer) => {
                    split_layer.execute(
                        &mut self.intermediates,
                        &self.inputs,
                        &self.initializers,
                    )?;
                }
                PlanNode::If(if_layer) => {
                    if_layer.execute(&mut self.intermediates, &self.inputs, &self.initializers)?;
                }
                PlanNode::TopK(topk_layer) => {
                    topk_layer.execute(
                        &mut self.intermediates,
                        &self.inputs,
                        &self.initializers,
                    )?;
                }
                PlanNode::Scan(scan_layer) => {
                    scan_layer.execute(
                        &mut self.intermediates,
                        &self.inputs,
                        &self.initializers,
                    )?;
                }
                #[cfg(feature = "xnnpack")]
                PlanNode::XnnpackSubgraph(sg) => {
                    sg.execute(&mut self.intermediates, &self.inputs, &self.initializers)?;
                }
            }
        }

        for name in output_names {
            let src = self
                .intermediates
                .get(name)
                .or_else(|| self.inputs.get(name))
                .or_else(|| self.initializers.get(name));
            if let Some(src) = src {
                let dst = outputs.entry(name.clone()).or_default();
                dst.copy_from(src);
            }
        }

        Ok(())
    }

    pub fn input_sizes(&self) -> HashMap<String, Dims> {
        let shape_map = self.current_plan.map(|idx| &self.plan_cache[idx].shape_map);
        self.input_names
            .iter()
            .filter_map(|name| {
                shape_map
                    .and_then(|sm| sm.get(name))
                    .map(|sl| (name.clone(), sl.dims.clone()))
            })
            .collect()
    }

    pub fn value(&self, name: &str) -> Option<&Tensor> {
        self.inputs
            .get(name)
            .or_else(|| self.initializers.get(name))
            .or_else(|| self.intermediates.get(name))
    }

    /// Returns a mutable slice to the f32 data of an input tensor, reusing
    /// the existing allocation when possible. The tensor is resized to match
    /// `dims`; no allocation occurs if the capacity is already sufficient.
    pub fn input_floats_mut(&mut self, name: &str, dims: Dims) -> Result<&mut [f32]> {
        let numel: usize = dims.iter().product();
        let tensor = self
            .inputs
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
        self.current_plan
            .map(|idx| {
                self.plan_cache[idx]
                    .shape_map
                    .iter()
                    .map(|(k, v)| (k.clone(), v.dims.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }
}
