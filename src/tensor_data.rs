use anyhow::Context;
use prost::Message;
use smallvec::SmallVec;

use crate::DType;
use crate::ONNX_DOUBLE;
use crate::ONNX_INT8;
use crate::ONNX_INT32;
use crate::ONNX_INT64;
use crate::ONNX_STRING;
use crate::ONNX_UINT8;
use crate::Result;
use crate::onnx::TensorProto;

pub type Dims = SmallVec<[usize; 8]>;

/// Data layout for spatial tensors.
///
/// ONNX convention is NCHW (batch, channels, height, width).
/// graph_opt may insert LayoutTranspose nodes that convert to NHWC for XNNPACK.
/// `Unknown` is used for non-spatial tensors (1D, 2D) or after rank-changing ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    #[default]
    NCHW,
    NHWC,
    Unknown,
}

/// Shape + layout metadata for a tensor, used in shape_map during plan building.
#[derive(Debug, Clone)]
pub struct ShapeLayout {
    pub dims: Dims,
    pub layout: Layout,
}

impl ShapeLayout {
    pub fn new(dims: Dims, layout: Layout) -> Self {
        Self { dims, layout }
    }

    pub fn nchw(dims: Dims) -> Self {
        Self {
            dims,
            layout: Layout::NCHW,
        }
    }

    pub fn unknown(dims: Dims) -> Self {
        Self {
            dims,
            layout: Layout::Unknown,
        }
    }
}

#[macro_export]
macro_rules! dims {
    ($elem:expr; $n:expr) => { smallvec::smallvec![$elem; $n] };
    ($($x:expr),* $(,)?) => { smallvec::smallvec![$($x),*] };
}

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Strings(Vec<Vec<u8>>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Dims,
    pub data: TensorData,
    pub layout: Layout,
}

impl Tensor {
    pub fn new(dims: Dims, data: Vec<f32>) -> Self {
        Self {
            dims,
            data: TensorData::F32(data),
            layout: Layout::NCHW,
        }
    }

    pub fn new_i64(dims: Dims, data: Vec<i64>) -> Self {
        Self {
            dims,
            data: TensorData::I64(data),
            layout: Layout::NCHW,
        }
    }

    pub fn new_strings(dims: Dims, data: Vec<Vec<u8>>) -> Self {
        Self {
            dims,
            data: TensorData::Strings(data),
            layout: Layout::NCHW,
        }
    }

    pub fn dtype(&self) -> DType {
        match &self.data {
            TensorData::F32(_) => DType::Float,
            TensorData::I64(_) => DType::Int64,
            TensorData::Strings(_) => DType::String,
        }
    }

    pub fn floats(&self) -> anyhow::Result<&[f32]> {
        match &self.data {
            TensorData::F32(buf) => Ok(buf),
            _ => anyhow::bail!("expected float tensor, got {:?}", self.dtype()),
        }
    }

    pub fn ints(&self) -> anyhow::Result<&[i64]> {
        match &self.data {
            TensorData::I64(buf) => Ok(buf),
            _ => anyhow::bail!("expected int64 tensor, got {:?}", self.dtype()),
        }
    }

    pub fn strings(&self) -> anyhow::Result<&[Vec<u8>]> {
        match &self.data {
            TensorData::Strings(buf) => Ok(buf),
            _ => anyhow::bail!("expected string tensor, got {:?}", self.dtype()),
        }
    }

    pub fn into_f32_vec(self) -> anyhow::Result<Vec<f32>> {
        match self.data {
            TensorData::F32(buf) => Ok(buf),
            TensorData::I64(buf) => Ok(buf.iter().map(|&v| v as f32).collect()),
            TensorData::Strings(_) => anyhow::bail!("string tensor not supported here"),
        }
    }

    pub fn into_i64_vec(self) -> anyhow::Result<Vec<i64>> {
        match self.data {
            TensorData::F32(buf) => Ok(buf.iter().map(|&v| v as i64).collect()),
            TensorData::I64(buf) => Ok(buf),
            TensorData::Strings(_) => anyhow::bail!("string tensor not supported here"),
        }
    }

    pub fn f32_at(&self, idx: usize) -> anyhow::Result<f32> {
        match &self.data {
            TensorData::F32(buf) => Ok(buf[idx]),
            TensorData::I64(buf) => Ok(buf[idx] as f32),
            TensorData::Strings(_) => anyhow::bail!("string tensor not supported here"),
        }
    }

    pub fn i64_at(&self, idx: usize) -> anyhow::Result<i64> {
        match &self.data {
            TensorData::F32(buf) => Ok(buf[idx] as i64),
            TensorData::I64(buf) => Ok(buf[idx]),
            TensorData::Strings(_) => anyhow::bail!("string tensor not supported here"),
        }
    }

    pub fn from_proto(proto: &TensorProto) -> Result<Self> {
        let dims: Dims = proto.dims.iter().map(|&d| d as usize).collect();
        if proto.data_type == ONNX_STRING {
            let data: Vec<Vec<u8>> = proto.string_data.iter().map(|s| s.to_vec()).collect();
            Ok(Self::new_strings(dims, data))
        } else if proto.data_type == ONNX_INT32 || proto.data_type == ONNX_INT64 {
            let data = extract_int_data(proto)?;
            Ok(Self::new_i64(dims, data))
        } else {
            let data = extract_float_data(proto)?;
            Ok(Self::new(dims, data))
        }
    }

    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self> {
        let proto = TensorProto::decode(bytes).context("decoding tensor proto")?;
        Self::from_proto(&proto)
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn as_mut_f32(&mut self, len: usize) -> &mut Vec<f32> {
        if !matches!(self.data, TensorData::F32(_)) {
            self.data = TensorData::F32(vec![0.0; len]);
        }
        match &mut self.data {
            TensorData::F32(buf) => {
                buf.resize(len, 0.0);
                buf
            }
            _ => unreachable!(),
        }
    }

    pub fn as_mut_i64(&mut self, len: usize) -> &mut Vec<i64> {
        if !matches!(self.data, TensorData::I64(_)) {
            self.data = TensorData::I64(vec![0; len]);
        }
        match &mut self.data {
            TensorData::I64(buf) => {
                buf.resize(len, 0);
                buf
            }
            _ => unreachable!(),
        }
    }

    pub fn set_dims(&mut self, dims: &[usize]) {
        self.dims.clear();
        self.dims.extend_from_slice(dims);
    }

    pub fn copy_from(&mut self, other: &Tensor) {
        self.dims.clone_from(&other.dims);
        self.layout = other.layout;
        match &other.data {
            TensorData::F32(src) => {
                let dst = self.as_mut_f32(src.len());
                dst.copy_from_slice(src);
            }
            TensorData::I64(src) => {
                let dst = self.as_mut_i64(src.len());
                dst.copy_from_slice(src);
            }
            TensorData::Strings(src) => {
                self.data = TensorData::Strings(src.clone());
            }
        }
    }

    pub fn copy_cast_f32(&mut self, src: &Tensor) -> anyhow::Result<()> {
        self.dims.clone_from(&src.dims);
        self.layout = src.layout;
        let len = src.numel();
        let buf = self.as_mut_f32(len);
        match &src.data {
            TensorData::F32(s) => {
                buf[..len].copy_from_slice(&s[..len]);
            }
            TensorData::I64(s) => {
                for i in 0..len {
                    buf[i] = s[i] as f32;
                }
            }
            TensorData::Strings(_) => anyhow::bail!("string tensor not supported here"),
        }
        Ok(())
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            dims: Dims::new(),
            data: TensorData::F32(vec![]),
            layout: Layout::NCHW,
        }
    }
}

fn extract_float_data(tensor: &TensorProto) -> Result<Vec<f32>> {
    let dtype = tensor.data_type;

    if !tensor.raw_data.is_empty() {
        return match dtype {
            ONNX_UINT8 => Ok(tensor.raw_data.iter().map(|&b| b as f32).collect()),
            ONNX_INT8 => Ok(tensor.raw_data.iter().map(|&b| (b as i8) as f32).collect()),
            ONNX_DOUBLE => Ok(tensor
                .raw_data
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect()),
            _ => Ok(tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
        };
    }
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(tensor.int32_data.iter().map(|&v| v as f32).collect());
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor.int64_data.iter().map(|&v| v as f32).collect());
    }
    Ok(vec![])
}

fn extract_int_data(tensor: &TensorProto) -> Result<Vec<i64>> {
    if !tensor.raw_data.is_empty() {
        return match tensor.data_type {
            ONNX_INT32 => Ok(tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64)
                .collect()),
            ONNX_INT64 => Ok(tensor
                .raw_data
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect()),
            _ => unreachable!(),
        };
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor.int64_data.clone());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(tensor.int32_data.iter().map(|&v| v as i64).collect());
    }
    Ok(vec![])
}
