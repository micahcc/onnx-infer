use std::collections::HashMap;

use crate::InferenceError;
use crate::Result;
use crate::Tensor;
use crate::get_attr_ints;
use crate::get_tensor;
use crate::onnx::NodeProto;

pub fn exec_maxpool(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;

    let kernel_shape = get_attr_ints(node, "kernel_shape")
        .ok_or_else(|| InferenceError::InvalidModel("MaxPool missing kernel_shape".into()))?;
    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let n = input.dims[0];
    let c = input.dims[1];
    let h_in = input.dims[2];
    let w_in = input.dims[3];

    let kh = kernel_shape[0] as usize;
    let kw = kernel_shape[1] as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let ph_begin = pads[0] as usize;
    let pw_begin = pads[1] as usize;

    let h_out = (h_in + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
    let w_out = (w_in + pads[1] as usize + pads[3] as usize - kw) / sw + 1;

    let mut output = vec![f32::NEG_INFINITY; n * c * h_out * w_out];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * sh + fh;
                            let iw = ow * sw + fw;
                            if ih >= ph_begin
                                && iw >= pw_begin
                                && ih - ph_begin < h_in
                                && iw - pw_begin < w_in
                            {
                                let ih = ih - ph_begin;
                                let iw = iw - pw_begin;
                                let idx = ((batch * c + ch) * h_in + ih) * w_in + iw;
                                max_val = max_val.max(input.data[idx]);
                            }
                        }
                    }
                    let out_idx = ((batch * c + ch) * h_out + oh) * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    values.insert(
        node.output[0].clone(),
        Tensor::new(vec![n, c, h_out, w_out], output),
    );
    Ok(())
}

pub fn exec_global_avg_pool(node: &NodeProto, values: &mut HashMap<String, Tensor>) -> Result<()> {
    let input = get_tensor(values, &node.input[0])?;
    // Input: [N, C, H, W] -> Output: [N, C, 1, 1]
    let n = input.dims[0];
    let c = input.dims[1];
    let spatial: usize = input.dims[2..].iter().product();

    let mut output = vec![0.0f32; n * c];
    for batch in 0..n {
        for ch in 0..c {
            let offset = (batch * c + ch) * spatial;
            let sum: f32 = input.data[offset..offset + spatial].iter().sum();
            output[batch * c + ch] = sum / spatial as f32;
        }
    }

    let mut out_dims = vec![n, c];
    out_dims.resize(input.dims.len(), 1);
    values.insert(node.output[0].clone(), Tensor::new(out_dims, output));
    Ok(())
}
