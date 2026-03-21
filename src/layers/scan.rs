use std::collections::HashMap;
use std::collections::HashSet;

use anyhow::Context;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx_ir::Graph;

pub struct Scan {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    body: Graph,
    num_scan_inputs: usize,
    scan_input_directions: Vec<i64>,
    scan_output_directions: Vec<i64>,
    plan: Option<Plan>,
    values: HashMap<String, Tensor>,
    outer_refs: Vec<String>,
    state_in_names: Vec<String>,
    state_out_names: Vec<String>,
    scan_in_names: Vec<String>,
    scan_out_names: Vec<String>,
    state: Vec<Tensor>,
    accum_f32: Vec<Vec<f32>>,
    accum_i64: Vec<Vec<i64>>,
    accum_elem_dims: Vec<Dims>,
    accum_dtypes: Vec<Option<DType>>,
}

impl Scan {
    pub fn new(
        inputs: Vec<String>,
        outputs: Vec<String>,
        body: Graph,
        num_scan_inputs: usize,
        scan_input_directions: Vec<i64>,
        scan_output_directions: Vec<i64>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
            plan: None,
            values: HashMap::new(),
            outer_refs: Vec::new(),
            state_in_names: Vec::new(),
            state_out_names: Vec::new(),
            scan_in_names: Vec::new(),
            scan_out_names: Vec::new(),
            state: Vec::new(),
            accum_f32: Vec::new(),
            accum_i64: Vec::new(),
            accum_elem_dims: Vec::new(),
            accum_dtypes: Vec::new(),
        }
    }

    fn init(&mut self, outer_values: &HashMap<String, Tensor>) -> Result<()> {
        let num_state = self.inputs.len() - self.num_scan_inputs;
        let num_scan_out = self.body.outputs.len() - num_state;

        self.state_in_names = (0..num_state)
            .map(|j| self.body.inputs[j].name.clone())
            .collect();
        self.state_out_names = (0..num_state)
            .map(|j| self.body.outputs[j].name.clone())
            .collect();
        self.scan_in_names = (0..self.num_scan_inputs)
            .map(|j| self.body.inputs[num_state + j].name.clone())
            .collect();
        self.scan_out_names = (0..num_scan_out)
            .map(|j| self.body.outputs[num_state + j].name.clone())
            .collect();

        // Outer references
        let mut body_local: HashSet<&str> = HashSet::new();
        for inp in &self.body.inputs {
            body_local.insert(&inp.name);
        }
        for name in self.body.initializers.keys() {
            body_local.insert(name);
        }
        for node in &self.body.nodes {
            for out in &node.outputs {
                body_local.insert(out);
            }
        }
        let mut outer_ref_set: HashSet<String> = HashSet::new();
        for node in &self.body.nodes {
            for inp in &node.inputs {
                if !inp.is_empty() && !body_local.contains(inp.as_str()) {
                    outer_ref_set.insert(inp.clone());
                }
            }
        }
        self.outer_refs = outer_ref_set.into_iter().collect();

        // Type hints
        let mut type_hints: HashMap<String, DType> = HashMap::new();
        for (j, name) in self.state_in_names.iter().enumerate() {
            if let Some(src) = outer_values.get(&self.inputs[j]) {
                type_hints.insert(name.clone(), src.dtype());
            }
        }
        for (j, name) in self.scan_in_names.iter().enumerate() {
            if let Some(src) = outer_values.get(&self.inputs[num_state + j]) {
                type_hints.insert(name.clone(), src.dtype());
            }
        }
        for name in &self.outer_refs {
            if let Some(t) = outer_values.get(name) {
                type_hints.insert(name.clone(), t.dtype());
            }
        }

        // Build plan with shape hints for scan inputs (sliced shapes)
        let num_state = self.state_in_names.len();
        let mut shape_hints: HashMap<String, Dims> = HashMap::new();
        for (j, name) in self.state_in_names.iter().enumerate() {
            if let Some(src) = outer_values.get(&self.inputs[j]) {
                shape_hints.insert(name.clone(), src.dims.clone());
            }
        }
        for (j, name) in self.scan_in_names.iter().enumerate() {
            if let Some(src) = outer_values.get(&self.inputs[num_state + j]) {
                if src.dims.len() > 1 {
                    let sliced: Dims = src.dims[1..].iter().copied().collect();
                    shape_hints.insert(name.clone(), sliced);
                }
            }
        }

        let mut plan = Plan::build_with_types(&self.body, &shape_hints, &type_hints)?;

        for (k, v) in std::mem::take(&mut plan.initializers) {
            self.values.insert(k, v);
        }
        for (k, v) in std::mem::take(&mut plan.tensor_pool) {
            self.values.insert(k, v);
        }
        for name in &self.outer_refs {
            if let Some(t) = outer_values.get(name) {
                self.values.insert(name.clone(), t.clone());
            }
        }

        for name in &self.state_in_names {
            if !self.values.contains_key(name) {
                self.values.insert(name.clone(), Tensor::default());
            }
        }
        for name in &self.scan_in_names {
            if !self.values.contains_key(name) {
                self.values.insert(name.clone(), Tensor::default());
            }
        }

        self.plan = Some(plan);
        self.state = (0..num_state).map(|_| Tensor::default()).collect();
        let num_scan_out = self.scan_out_names.len();
        self.accum_f32 = vec![Vec::new(); num_scan_out];
        self.accum_i64 = vec![Vec::new(); num_scan_out];
        self.accum_elem_dims = vec![Dims::new(); num_scan_out];
        self.accum_dtypes = vec![None; num_scan_out];

        Ok(())
    }

    pub fn execute(&mut self, outer_values: &mut HashMap<String, Tensor>) -> Result<()> {
        if self.plan.is_none() {
            self.init(outer_values)?;
        }

        let num_state = self.state_in_names.len();

        let seq_len = if self.num_scan_inputs > 0 {
            let scan_in = get_tensor(outer_values, &self.inputs[num_state])?;
            scan_in.dims[0]
        } else {
            anyhow::bail!("Scan requires at least one scan input");
        };

        for (j, _name) in self.state_in_names.iter().enumerate() {
            let src = get_tensor(outer_values, &self.inputs[j])?;
            self.state[j].copy_from(src);
        }

        for name in &self.outer_refs {
            if let Some(outer) = outer_values.get(name) {
                if let Some(body) = self.values.get_mut(name) {
                    body.copy_from(outer);
                }
            }
        }

        for j in 0..self.scan_out_names.len() {
            self.accum_f32[j].clear();
            self.accum_i64[j].clear();
            self.accum_dtypes[j] = None;
        }

        struct ScanInputInfo {
            data_f32: Vec<f32>,
            data_i64: Vec<i64>,
            dtype: DType,
            slice_dims: Dims,
            slice_size: usize,
            reverse: bool,
        }
        let mut scan_inputs: Vec<ScanInputInfo> = Vec::with_capacity(self.num_scan_inputs);
        for j in 0..self.num_scan_inputs {
            let t = get_tensor(outer_values, &self.inputs[num_state + j])?;
            let slice_dims: Dims = t.dims[1..].iter().copied().collect();
            let slice_size: usize = slice_dims.iter().product();
            let reverse = self.scan_input_directions.get(j).copied().unwrap_or(0) != 0;
            scan_inputs.push(ScanInputInfo {
                data_f32: if t.dtype() == DType::Float {
                    t.floats().context("in Scan layer")?.to_vec()
                } else {
                    Vec::new()
                },
                data_i64: if t.dtype() == DType::Int64 {
                    t.ints().context("in Scan layer")?.to_vec()
                } else {
                    Vec::new()
                },
                dtype: t.dtype(),
                slice_dims,
                slice_size,
                reverse,
            });
        }

        let Scan {
            plan,
            values,
            state,
            state_in_names,
            state_out_names,
            scan_in_names,
            scan_out_names,
            scan_output_directions,
            accum_f32,
            accum_i64,
            accum_elem_dims,
            accum_dtypes,
            ..
        } = self;
        let plan = plan.as_mut().unwrap();

        for i in 0..seq_len {
            for (j, name) in state_in_names.iter().enumerate() {
                let (key, mut dst) = values
                    .remove_entry(name.as_str())
                    .unwrap_or_else(|| (name.clone(), Tensor::default()));
                dst.copy_from(&state[j]);
                values.insert(key, dst);
            }

            for (j, name) in scan_in_names.iter().enumerate() {
                let info = &scan_inputs[j];
                let t_idx = if info.reverse { seq_len - 1 - i } else { i };
                let off = t_idx * info.slice_size;

                let (key, mut dst) = values
                    .remove_entry(name.as_str())
                    .unwrap_or_else(|| (name.clone(), Tensor::default()));
                match info.dtype {
                    DType::Float => {
                        let buf = dst.as_mut_f32(info.slice_size);
                        buf.copy_from_slice(&info.data_f32[off..off + info.slice_size]);
                    }
                    DType::Int64 => {
                        let buf = dst.as_mut_i64(info.slice_size);
                        buf.copy_from_slice(&info.data_i64[off..off + info.slice_size]);
                    }
                    DType::String => unreachable!("strings not supported"),
                }
                dst.set_dims(&info.slice_dims);
                values.insert(key, dst);
            }

            for node in &mut plan.nodes {
                match node {
                    PlanNode::Single { output, layer } => {
                        if output.is_empty() {
                            continue;
                        }
                        let (key, mut out) = values
                            .remove_entry(output.as_str())
                            .unwrap_or_else(|| (output.clone(), Tensor::default()));
                        let result = layer.execute(values, &mut out);
                        values.insert(key, out);
                        result?;
                    }
                    PlanNode::Loop(loop_layer) => loop_layer.execute(values)?,
                    PlanNode::Split(split_layer) => split_layer.execute(values)?,
                    PlanNode::If(if_layer) => if_layer.execute(values)?,
                    PlanNode::TopK(topk_layer) => topk_layer.execute(values)?,
                    PlanNode::Scan(scan_layer) => scan_layer.execute(values)?,
                    #[cfg(feature = "xnnpack")]
                    PlanNode::XnnpackSubgraph(sg) => sg.execute(values)?,
                }
            }

            for (j, name) in state_out_names.iter().enumerate() {
                if let Some(t) = values.get(name) {
                    state[j].copy_from(t);
                }
            }

            for (j, name) in scan_out_names.iter().enumerate() {
                if let Some(t) = values.get(name) {
                    if accum_dtypes[j].is_none() {
                        accum_dtypes[j] = Some(t.dtype());
                        accum_elem_dims[j].clear();
                        accum_elem_dims[j].extend_from_slice(&t.dims);
                    }
                    match t.dtype() {
                        DType::Float => {
                            accum_f32[j].extend_from_slice(t.floats().context("in Scan layer")?)
                        }
                        DType::Int64 => {
                            accum_i64[j].extend_from_slice(t.ints().context("in Scan layer")?)
                        }
                        DType::String => unreachable!("strings not supported"),
                    }
                }
            }
        }

        // Write state outputs
        let mut out_idx = 0;
        let outputs = &self.outputs;
        for s in state.iter().take(num_state) {
            if out_idx < outputs.len() && !outputs[out_idx].is_empty() {
                let (key, mut outer) = outer_values
                    .remove_entry(outputs[out_idx].as_str())
                    .unwrap_or_else(|| (outputs[out_idx].clone(), Tensor::default()));
                outer.copy_from(s);
                outer_values.insert(key, outer);
            }
            out_idx += 1;
        }

        // Write scan outputs
        let num_scan_out = scan_out_names.len();
        for j in 0..num_scan_out {
            if out_idx < outputs.len() && !outputs[out_idx].is_empty() {
                let reverse_out = scan_output_directions.get(j).copied().unwrap_or(0) != 0;
                let (key, mut outer) = outer_values
                    .remove_entry(outputs[out_idx].as_str())
                    .unwrap_or_else(|| (outputs[out_idx].clone(), Tensor::default()));

                if accum_dtypes[j].is_none() || seq_len == 0 {
                    outer.set_dims(&[0]);
                    outer.as_mut_f32(0);
                } else {
                    let elem = &accum_elem_dims[j];
                    let mut out_dims = Dims::new();
                    out_dims.push(seq_len);
                    out_dims.extend_from_slice(elem);
                    let elem_size: usize = elem.iter().product();

                    match accum_dtypes[j].unwrap() {
                        DType::Float => {
                            let buf = outer.as_mut_f32(accum_f32[j].len());
                            if reverse_out {
                                for t in 0..seq_len {
                                    let src_off = (seq_len - 1 - t) * elem_size;
                                    let dst_off = t * elem_size;
                                    buf[dst_off..dst_off + elem_size].copy_from_slice(
                                        &accum_f32[j][src_off..src_off + elem_size],
                                    );
                                }
                            } else {
                                buf.copy_from_slice(&accum_f32[j]);
                            }
                        }
                        DType::Int64 => {
                            let buf = outer.as_mut_i64(accum_i64[j].len());
                            if reverse_out {
                                for t in 0..seq_len {
                                    let src_off = (seq_len - 1 - t) * elem_size;
                                    let dst_off = t * elem_size;
                                    buf[dst_off..dst_off + elem_size].copy_from_slice(
                                        &accum_i64[j][src_off..src_off + elem_size],
                                    );
                                }
                            } else {
                                buf.copy_from_slice(&accum_i64[j]);
                            }
                        }
                        DType::String => unreachable!("strings not supported"),
                    }
                    outer.set_dims(&out_dims);
                }
                outer_values.insert(key, outer);
            }
            out_idx += 1;
        }

        Ok(())
    }
}
