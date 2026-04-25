use std::collections::HashMap;

use crate::DType;
use crate::Dims;
use crate::Result;
use crate::Tensor;
use crate::layers::OpType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    Float,
    Uint8,
    Int8,
    Int32,
    Int64,
    String,
    Double,
    Bool,
    Unknown(i32),
}

impl ElemType {
    pub fn from_onnx(v: i32) -> Self {
        match v {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            11 => Self::Double,
            other => Self::Unknown(other),
        }
    }

    pub fn to_dtype(self) -> DType {
        match self {
            Self::Int32 | Self::Int64 => DType::Int64,
            Self::String => DType::String,
            _ => DType::Float,
        }
    }

    pub fn is_int(self) -> bool {
        matches!(self, Self::Int32 | Self::Int64)
    }
}

#[derive(Debug, Clone)]
pub enum Attr {
    Int(i64),
    Float(f32),
    String(String),
    Tensor(Tensor),
    Graph(Box<Graph>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<Vec<u8>>),
}

#[derive(Debug, Clone)]
pub struct Attrs(pub HashMap<String, Attr>);

impl Attrs {
    pub fn get(&self, name: &str) -> Option<&Attr> {
        self.0.get(name)
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.0.get(name)? {
            Attr::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.0.get(name)? {
            Attr::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_ints(&self, name: &str) -> Option<Vec<i64>> {
        match self.0.get(name)? {
            Attr::Ints(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn get_string(&self, name: &str) -> Option<String> {
        match self.0.get(name)? {
            Attr::String(v) if !v.is_empty() => Some(v.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op_type: OpType,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Attrs,
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub elem_type: ElemType,
    pub shape: Option<Dims>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub initializers: HashMap<String, Tensor>,
    pub opset_version: i64,
}

pub fn convert_graph(graph: &crate::onnx::GraphProto) -> Result<Graph> {
    convert_graph_with_opset(graph, 0)
}

pub fn convert_graph_with_opset(
    graph: &crate::onnx::GraphProto,
    opset_version: i64,
) -> Result<Graph> {
    let mut initializers = HashMap::new();
    for init in &graph.initializer {
        if !init.name.is_empty() {
            initializers.insert(init.name.clone(), Tensor::from_proto(init)?);
        }
    }

    let inputs = graph.input.iter().map(convert_value_info).collect();
    let outputs = graph.output.iter().map(convert_value_info).collect();
    let nodes = graph.node.iter().map(convert_node).collect::<Result<_>>()?;

    Ok(Graph {
        nodes,
        inputs,
        outputs,
        initializers,
        opset_version,
    })
}

fn convert_value_info(vi: &crate::onnx::ValueInfoProto) -> ValueInfo {
    let name = vi.name.clone();
    let (elem_type, shape) = vi
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .map(|v| match v {
            crate::onnx::type_proto::Value::TensorType(tt) => {
                let shape = tt.shape.as_ref().and_then(|s| {
                    if s.dim.is_empty() {
                        return None;
                    }
                    let mut dims = Dims::new();
                    for d in &s.dim {
                        match &d.value {
                            Some(crate::onnx::tensor_shape_proto::dimension::Value::DimValue(
                                v,
                            )) if *v > 0 => {
                                dims.push(*v as usize);
                            }
                            _ => dims.push(0),
                        }
                    }
                    Some(dims)
                });
                (ElemType::from_onnx(tt.elem_type), shape)
            }
            _ => (ElemType::Unknown(0), None),
        })
        .unwrap_or((ElemType::Unknown(0), None));
    ValueInfo {
        name,
        elem_type,
        shape,
    }
}

fn convert_node(node: &crate::onnx::NodeProto) -> Result<Node> {
    let op_type =
        OpType::parse(&node.op_type).map_err(|s| anyhow::anyhow!("unsupported operator: {s}"))?;
    let attrs = node
        .attribute
        .iter()
        .map(convert_attr)
        .collect::<Result<_>>()?;
    Ok(Node {
        op_type,
        name: node.name.clone(),
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        attrs: Attrs(attrs),
    })
}

fn convert_attr(attr: &crate::onnx::AttributeProto) -> Result<(String, Attr)> {
    let name = attr.name.clone();
    let val = match attr.r#type {
        1 => Attr::Float(attr.f),
        2 => Attr::Int(attr.i),
        3 => Attr::String(String::from_utf8_lossy(&attr.s).to_string()),
        4 => {
            let t = attr.t.as_ref().ok_or_else(|| {
                anyhow::anyhow!("Attribute '{name}' has tensor type but no tensor")
            })?;
            Attr::Tensor(Tensor::from_proto(t)?)
        }
        5 => {
            let g = attr
                .g
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Attribute '{name}' has graph type but no graph"))?;
            Attr::Graph(Box::new(convert_graph(g)?))
        }
        6 => Attr::Floats(attr.floats.clone()),
        7 => Attr::Ints(attr.ints.clone()),
        8 => Attr::Strings(attr.strings.clone()),
        _ => {
            // type 0 (UNDEFINED) or unknown: detect from populated fields
            if let Some(t) = &attr.t {
                Attr::Tensor(Tensor::from_proto(t)?)
            } else if let Some(g) = &attr.g {
                Attr::Graph(Box::new(convert_graph(g)?))
            } else if !attr.ints.is_empty() {
                Attr::Ints(attr.ints.clone())
            } else if !attr.floats.is_empty() {
                Attr::Floats(attr.floats.clone())
            } else if !attr.strings.is_empty() {
                Attr::Strings(attr.strings.clone())
            } else if !attr.s.is_empty() {
                Attr::String(String::from_utf8_lossy(&attr.s).to_string())
            } else if attr.f != 0.0 {
                Attr::Float(attr.f)
            } else {
                Attr::Int(attr.i)
            }
        }
    };
    Ok((name, val))
}
