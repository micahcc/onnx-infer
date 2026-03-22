use std::collections::HashMap;
use std::collections::HashSet;

use anyhow::Context;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::dims;
use crate::get_tensor;
use crate::layers::Plan;
use crate::layers::PlanNode;
use crate::onnx_ir::Graph;

pub struct Loop {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    body: Graph,
    plan: Option<Plan>,
    values: HashMap<String, Tensor>,
    outer_refs: Vec<String>,
    iter_name: String,
    cond_name: String,
    cond_out_name: String,
    carried_in_names: Vec<String>,
    carried_out_names: Vec<String>,
    scan_out_names: Vec<String>,
    // Steady-state types for carried inputs (may differ from initial outer values)
    carried_types: Vec<DType>,
    carried: Vec<Tensor>,
    scan_f32: Vec<Vec<f32>>,
    scan_i64: Vec<Vec<i64>>,
    scan_elem_dims: Vec<Dims>,
    scan_dtypes: Vec<Option<DType>>,
}

impl Loop {
    pub fn new(inputs: Vec<String>, outputs: Vec<String>, mut body: Graph) -> Self {
        crate::graph_opt::optimize(&mut body);
        Self {
            inputs,
            outputs,
            body,
            plan: None,
            values: HashMap::new(),
            outer_refs: Vec::new(),
            iter_name: String::new(),
            cond_name: String::new(),
            cond_out_name: String::new(),
            carried_in_names: Vec::new(),
            carried_out_names: Vec::new(),
            scan_out_names: Vec::new(),
            carried_types: Vec::new(),
            carried: Vec::new(),
            scan_f32: Vec::new(),
            scan_i64: Vec::new(),
            scan_elem_dims: Vec::new(),
            scan_dtypes: Vec::new(),
        }
    }

    fn init(&mut self, outer_values: &HashMap<String, Tensor>) -> Result<()> {
        let num_carried = self.inputs.len() - 2;
        let num_scan = self.body.outputs.len() - 1 - num_carried;

        // Pre-compute names
        self.iter_name = self.body.inputs[0].name.clone();
        self.cond_name = self.body.inputs[1].name.clone();
        self.cond_out_name = self.body.outputs[0].name.clone();
        self.carried_in_names = (0..num_carried)
            .map(|j| self.body.inputs[j + 2].name.clone())
            .collect();
        self.carried_out_names = (0..num_carried)
            .map(|j| self.body.outputs[j + 1].name.clone())
            .collect();
        self.scan_out_names = (0..num_scan)
            .map(|j| self.body.outputs[1 + num_carried + j].name.clone())
            .collect();

        // Determine outer references
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

        // First pass: build type_hints from actual values and determine carried output types
        let mut type_hints: HashMap<String, DType> = HashMap::new();
        type_hints.insert(self.iter_name.clone(), DType::Int64);
        type_hints.insert(self.cond_name.clone(), DType::Float);
        for (j, name) in self.carried_in_names.iter().enumerate() {
            if let Some(src) = outer_values.get(&self.inputs[j + 2]) {
                type_hints.insert(name.clone(), src.dtype());
            }
        }
        for name in &self.outer_refs {
            if let Some(t) = outer_values.get(name) {
                type_hints.insert(name.clone(), t.dtype());
            }
        }

        // Build plan to determine steady-state types via type inference
        let probe = Plan::build_with_types(&self.body, &HashMap::new(), &type_hints)?;

        // Determine steady-state carried types from output inference
        let mut needs_rebuild = false;
        self.carried_types = Vec::with_capacity(num_carried);
        for j in 0..num_carried {
            let out_name = &self.carried_out_names[j];
            let in_name = &self.carried_in_names[j];
            let out_dt = probe
                .type_map
                .get(out_name)
                .copied()
                .unwrap_or(DType::Float);
            let in_dt = type_hints.get(in_name).copied().unwrap_or(DType::Float);
            self.carried_types.push(out_dt);
            if out_dt != in_dt {
                type_hints.insert(in_name.clone(), out_dt);
                needs_rebuild = true;
            }
        }

        // Rebuild with corrected types if any carried input/output types differed
        let plan = if needs_rebuild {
            Plan::build_with_types(&self.body, &HashMap::new(), &type_hints)?
        } else {
            probe
        };

        // Move initializers and tensor_pool into persistent values
        let mut plan = plan;
        for (k, v) in std::mem::take(&mut plan.initializers) {
            self.values.insert(k, v);
        }
        for (k, v) in std::mem::take(&mut plan.tensor_pool) {
            self.values.insert(k, v);
        }

        // Copy outer references
        for name in &self.outer_refs {
            if let Some(t) = outer_values.get(name) {
                self.values.insert(name.clone(), t.clone());
            }
        }

        // Ensure body inputs exist
        if !self.values.contains_key(&self.iter_name) {
            self.values
                .insert(self.iter_name.clone(), Tensor::new_i64(dims![], vec![0]));
        }
        if !self.values.contains_key(&self.cond_name) {
            self.values
                .insert(self.cond_name.clone(), Tensor::new(dims![], vec![1.0]));
        }
        for (j, name) in self.carried_in_names.iter().enumerate() {
            if !self.values.contains_key(name) {
                if let Some(src) = outer_values.get(&self.inputs[j + 2]) {
                    let mut t = src.clone();
                    // Cast to steady-state type if needed
                    if t.dtype() != self.carried_types[j] && self.carried_types[j] == DType::Float {
                        let mut casted = Tensor::default();
                        casted.copy_cast_f32(&t).context("in Loop layer")?;
                        t = casted;
                    }
                    self.values.insert(name.clone(), t);
                } else {
                    self.values.insert(name.clone(), Tensor::default());
                }
            }
        }

        self.plan = Some(plan);

        // Initialize scratch
        self.carried = (0..num_carried).map(|_| Tensor::default()).collect();
        self.scan_f32 = vec![Vec::new(); num_scan];
        self.scan_i64 = vec![Vec::new(); num_scan];
        self.scan_elem_dims = vec![Dims::new(); num_scan];
        self.scan_dtypes = vec![None; num_scan];

        Ok(())
    }

    pub fn execute(&mut self, outer_values: &mut HashMap<String, Tensor>) -> Result<()> {
        if self.plan.is_none() {
            self.init(outer_values)?;
        }

        let trip_tensor = get_tensor(outer_values, &self.inputs[0])?;
        let trip_count = trip_tensor
            .i64_at(0)
            .context("in Loop layer: reading trip count")? as usize;
        let num_carried = self.inputs.len() - 2;

        // Copy initial carried state, casting to steady-state type
        for j in 0..num_carried {
            let src = get_tensor(outer_values, &self.inputs[j + 2])?;
            if src.dtype() != self.carried_types[j] && self.carried_types[j] == DType::Float {
                self.carried[j]
                    .copy_cast_f32(src)
                    .context("in Loop layer")?;
            } else {
                self.carried[j].copy_from(src);
            }
        }

        // Update outer references
        for name in &self.outer_refs {
            if let Some(outer) = outer_values.get(name) {
                if let Some(body) = self.values.get_mut(name) {
                    body.copy_from(outer);
                }
            }
        }

        // Reset scan accumulators
        let num_scan = self.scan_out_names.len();
        for j in 0..num_scan {
            self.scan_f32[j].clear();
            self.scan_i64[j].clear();
            self.scan_dtypes[j] = None;
        }

        // Destructure for split borrows
        let Loop {
            plan,
            values,
            carried,
            carried_types,
            scan_f32,
            scan_i64,
            scan_elem_dims,
            scan_dtypes,
            outputs,
            iter_name,
            cond_name,
            cond_out_name,
            carried_in_names,
            carried_out_names,
            scan_out_names,
            ..
        } = self;
        let plan = plan.as_mut().unwrap();

        let mut actual_iters = 0usize;
        for i in 0..trip_count {
            // Update iteration counter
            {
                let iter_t = values.get_mut(iter_name).unwrap();
                let buf = iter_t.as_mut_i64(1);
                buf[0] = i as i64;
                iter_t.set_dims(&[]);
            }

            // Update condition
            {
                let cond_t = values.get_mut(cond_name).unwrap();
                let buf = cond_t.as_mut_f32(1);
                buf[0] = 1.0;
                cond_t.set_dims(&[]);
            }

            // Update carried state
            for (j, name) in carried_in_names.iter().enumerate() {
                let (key, mut dst) = values
                    .remove_entry(name.as_str())
                    .unwrap_or_else(|| (name.clone(), Tensor::default()));
                dst.copy_from(&carried[j]);
                values.insert(key, dst);
            }

            // Execute body plan nodes
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
                    PlanNode::Loop(loop_layer) => {
                        loop_layer.execute(values)?;
                    }
                    PlanNode::Split(split_layer) => {
                        split_layer.execute(values)?;
                    }
                    PlanNode::If(if_layer) => {
                        if_layer.execute(values)?;
                    }
                    PlanNode::TopK(topk_layer) => {
                        topk_layer.execute(values)?;
                    }
                    PlanNode::Scan(scan_layer) => {
                        scan_layer.execute(values)?;
                    }
                    #[cfg(feature = "xnnpack")]
                    PlanNode::XnnpackSubgraph(sg) => {
                        sg.execute(values)?;
                    }
                }
            }

            // Check condition output
            let keep_going = if let Some(cond) = values.get(cond_out_name) {
                match cond.dtype() {
                    DType::Float => {
                        cond.floats()
                            .context("in Loop layer")?
                            .first()
                            .copied()
                            .unwrap_or(0.0)
                            != 0.0
                    }
                    DType::Int64 => {
                        cond.ints()
                            .context("in Loop layer")?
                            .first()
                            .copied()
                            .unwrap_or(0)
                            != 0
                    }
                    DType::String => unreachable!("strings not supported"),
                }
            } else {
                true
            };

            // Extract carried state and assert consistent types
            for (j, out_name) in carried_out_names.iter().enumerate() {
                if let Some(t) = values.get(out_name) {
                    if t.dtype() != carried_types[j] {
                        anyhow::bail!(
                            "Loop carried output '{}' changed type from {:?} to {:?} at iteration {}. \
                             Loop body outputs must have consistent types across iterations.",
                            out_name,
                            carried_types[j],
                            t.dtype(),
                            i
                        );
                    }
                    carried[j].copy_from(t);
                }
            }

            // Accumulate scan outputs and assert consistent types/shapes
            for (j, scan_name) in scan_out_names.iter().enumerate() {
                if let Some(t) = values.get(scan_name) {
                    if let Some(expected_dt) = scan_dtypes[j] {
                        if t.dtype() != expected_dt {
                            anyhow::bail!(
                                "Loop scan output '{}' changed type from {:?} to {:?} at iteration {}. \
                                 Loop body outputs must have consistent types across iterations.",
                                scan_name,
                                expected_dt,
                                t.dtype(),
                                i
                            );
                        }
                        if t.dims != scan_elem_dims[j] {
                            anyhow::bail!(
                                "Loop scan output '{}' changed shape from {:?} to {:?} at iteration {}. \
                                 Loop body outputs must have consistent shapes across iterations.",
                                scan_name,
                                scan_elem_dims[j],
                                t.dims,
                                i
                            );
                        }
                    } else {
                        scan_dtypes[j] = Some(t.dtype());
                        scan_elem_dims[j].clear();
                        scan_elem_dims[j].extend_from_slice(&t.dims);
                    }
                    match t.dtype() {
                        DType::Float => {
                            scan_f32[j].extend_from_slice(t.floats().context("in Loop layer")?)
                        }
                        DType::Int64 => {
                            scan_i64[j].extend_from_slice(t.ints().context("in Loop layer")?)
                        }
                        DType::String => unreachable!("strings not supported"),
                    }
                }
            }

            actual_iters += 1;
            if !keep_going {
                break;
            }
        }

        // Write carried outputs to outer values
        let mut out_idx = 0;
        for c in carried.iter().take(num_carried) {
            if out_idx < outputs.len() && !outputs[out_idx].is_empty() {
                let (key, mut outer) = outer_values
                    .remove_entry(outputs[out_idx].as_str())
                    .unwrap_or_else(|| (outputs[out_idx].clone(), Tensor::default()));
                outer.copy_from(c);
                outer_values.insert(key, outer);
            }
            out_idx += 1;
        }

        // Write scan outputs
        for j in 0..num_scan {
            if out_idx < outputs.len() && !outputs[out_idx].is_empty() {
                let (key, mut outer) = outer_values
                    .remove_entry(outputs[out_idx].as_str())
                    .unwrap_or_else(|| (outputs[out_idx].clone(), Tensor::default()));
                if scan_dtypes[j].is_none() || actual_iters == 0 {
                    outer.set_dims(&[0]);
                    outer.as_mut_f32(0);
                } else {
                    let elem = &scan_elem_dims[j];
                    let mut dims_buf = [0usize; 8];
                    dims_buf[0] = actual_iters;
                    for (k, &d) in elem.iter().enumerate() {
                        dims_buf[k + 1] = d;
                    }
                    let rank = elem.len() + 1;
                    match scan_dtypes[j].unwrap() {
                        DType::Float => {
                            let buf = outer.as_mut_f32(scan_f32[j].len());
                            buf.copy_from_slice(&scan_f32[j]);
                        }
                        DType::Int64 => {
                            let buf = outer.as_mut_i64(scan_i64[j].len());
                            buf.copy_from_slice(&scan_i64[j]);
                        }
                        DType::String => unreachable!("strings not supported"),
                    }
                    outer.set_dims(&dims_buf[..rank]);
                }
                outer_values.insert(key, outer);
            }
            out_idx += 1;
        }

        Ok(())
    }
}
