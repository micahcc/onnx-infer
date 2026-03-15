use prost::Message;

use crate::DType;
use crate::InferenceError;
use crate::ONNX_DOUBLE;
use crate::ONNX_INT32;
use crate::ONNX_INT64;
use crate::ONNX_INT8;
use crate::ONNX_UINT8;
use crate::Result;
use crate::onnx::TensorProto;

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    pub fn new(dims: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            dims,
            data: TensorData::F32(data),
        }
    }

    pub fn new_i64(dims: Vec<usize>, data: Vec<i64>) -> Self {
        Self {
            dims,
            data: TensorData::I64(data),
        }
    }

    pub fn dtype(&self) -> DType {
        match &self.data {
            TensorData::F32(_) => DType::Float,
            TensorData::I64(_) => DType::Int64,
        }
    }

    pub fn floats(&self) -> &[f32] {
        match &self.data {
            TensorData::F32(buf) => buf,
            TensorData::I64(_) => panic!("expected float tensor, got int64"),
        }
    }

    pub fn ints(&self) -> &[i64] {
        match &self.data {
            TensorData::I64(buf) => buf,
            TensorData::F32(_) => panic!("expected int64 tensor, got float"),
        }
    }

    pub fn into_f32_vec(self) -> Vec<f32> {
        match self.data {
            TensorData::F32(buf) => buf,
            TensorData::I64(buf) => buf.iter().map(|&v| v as f32).collect(),
        }
    }

    pub fn into_i64_vec(self) -> Vec<i64> {
        match self.data {
            TensorData::F32(buf) => buf.iter().map(|&v| v as i64).collect(),
            TensorData::I64(buf) => buf,
        }
    }

    pub fn f32_at(&self, idx: usize) -> f32 {
        match &self.data {
            TensorData::F32(buf) => buf[idx],
            TensorData::I64(buf) => buf[idx] as f32,
        }
    }

    pub fn i64_at(&self, idx: usize) -> i64 {
        match &self.data {
            TensorData::F32(buf) => buf[idx] as i64,
            TensorData::I64(buf) => buf[idx],
        }
    }

    pub fn from_proto(proto: &TensorProto) -> Result<Self> {
        let dims: Vec<usize> = proto.dims.iter().map(|&d| d as usize).collect();
        if proto.data_type == ONNX_INT32 || proto.data_type == ONNX_INT64 {
            let data = extract_int_data(proto)?;
            Ok(Self::new_i64(dims, data))
        } else {
            let data = extract_float_data(proto)?;
            Ok(Self::new(dims, data))
        }
    }

    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self> {
        let proto = TensorProto::decode(bytes).map_err(InferenceError::ParseError)?;
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

    pub fn data_replace_f32(&mut self, data: Vec<f32>) {
        self.data = TensorData::F32(data);
    }

    pub fn copy_from(&mut self, other: &Tensor) {
        self.dims.clone_from(&other.dims);
        match &other.data {
            TensorData::F32(src) => {
                let dst = self.as_mut_f32(src.len());
                dst.copy_from_slice(src);
            }
            TensorData::I64(src) => {
                let dst = self.as_mut_i64(src.len());
                dst.copy_from_slice(src);
            }
        }
    }

    pub fn copy_cast_f32(&mut self, src: &Tensor) {
        self.dims.clone_from(&src.dims);
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
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            dims: vec![],
            data: TensorData::F32(vec![]),
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
