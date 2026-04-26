use std::collections::HashMap;
use std::sync::Arc;

use crate::Tensor;

/// Read-only lookup over inputs, plan-folded constants, and initializers.
/// Priority: inputs > folded > initializers.
#[derive(Clone)]
pub struct Constants<'a> {
    pub inputs: &'a HashMap<String, Tensor>,
    pub folded: Arc<HashMap<String, Tensor>>,
    pub initializers: Arc<HashMap<String, Tensor>>,
}

impl<'a> Constants<'a> {
    pub fn empty() -> Constants<'static> {
        static EMPTY: std::sync::LazyLock<HashMap<String, crate::Tensor>> =
            std::sync::LazyLock::new(HashMap::new);
        Constants {
            inputs: &EMPTY,
            folded: Arc::new(HashMap::new()),
            initializers: Arc::new(HashMap::new()),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.inputs
            .get(name)
            .or_else(|| self.folded.get(name))
            .or_else(|| self.initializers.get(name))
    }
}
