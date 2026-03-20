use anyhow::Context;
use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::plan::execute_node;
use crate::onnx_ir::Graph;

pub struct If {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    then_branch: Graph,
    else_branch: Graph,
}

impl If {
    pub fn new(
        inputs: Vec<String>,
        outputs: Vec<String>,
        then_branch: Graph,
        else_branch: Graph,
    ) -> Self {
        Self {
            inputs,
            outputs,
            then_branch,
            else_branch,
        }
    }

    pub fn execute(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let cond = get_tensor(values, &self.inputs[0])?;
        let is_true = match cond.dtype() {
            DType::Float => cond.floats().context("in If layer")?.first().copied().unwrap_or(0.0) != 0.0,
            DType::Int64 => cond.ints().context("in If layer")?.first().copied().unwrap_or(0) != 0,
            DType::String => unreachable!("strings not supported"),
        };

        let branch = if is_true {
            &self.then_branch
        } else {
            &self.else_branch
        };

        // Copy initializers into values
        for (name, tensor) in &branch.initializers {
            if !values.contains_key(name) {
                values.insert(name.clone(), tensor.clone());
            }
        }

        for node in &branch.nodes {
            execute_node(node, values)?;
        }

        // Map branch outputs to If outputs
        for (i, out_name) in self.outputs.iter().enumerate() {
            if out_name.is_empty() {
                continue;
            }
            if let Some(branch_out) = branch.outputs.get(i) {
                if branch_out.name != *out_name {
                    if let Some(src) = values.remove(&branch_out.name) {
                        values.insert(out_name.clone(), src);
                    }
                }
            }
        }

        Ok(())
    }
}
