use std::collections::HashMap;

use crate::DType;
use crate::Result;
use crate::Tensor;
use crate::get_tensor;
use crate::layers::plan::execute_node;
use crate::onnx::GraphProto;

pub struct If {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    then_branch: GraphProto,
    else_branch: GraphProto,
}

impl If {
    pub fn new(
        inputs: Vec<String>,
        outputs: Vec<String>,
        then_branch: GraphProto,
        else_branch: GraphProto,
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
            DType::Float => cond.floats().first().copied().unwrap_or(0.0) != 0.0,
            DType::Int64 => cond.ints().first().copied().unwrap_or(0) != 0,
            DType::String => unreachable!("strings not supported"),
        };

        let branch = if is_true {
            &self.then_branch
        } else {
            &self.else_branch
        };

        // Execute the branch graph using the outer values
        // Copy initializers into values
        for init in &branch.initializer {
            if !init.name.is_empty()
                && !values.contains_key(&init.name)
                && let Ok(t) = Tensor::from_proto(init)
            {
                values.insert(init.name.clone(), t);
            }
        }

        for node in &branch.node {
            execute_node(node, values)?;
        }

        // Map branch outputs to If outputs
        for (i, out_name) in self.outputs.iter().enumerate() {
            if out_name.is_empty() {
                continue;
            }
            if let Some(branch_out) = branch.output.get(i)
                && branch_out.name != *out_name
                && let Some(src) = values.remove(&branch_out.name)
            {
                values.insert(out_name.clone(), src);
            }
        }

        Ok(())
    }
}
