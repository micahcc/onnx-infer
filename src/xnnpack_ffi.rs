//! Raw FFI bindings for the XNNPACK subgraph API.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;

// Opaque pointer types
pub type xnn_subgraph_t = *mut c_void;
pub type xnn_runtime_t = *mut c_void;
pub type pthreadpool_t = *mut c_void;

pub const XNN_INVALID_VALUE_ID: u32 = u32::MAX;
pub const XNN_VALUE_FLAG_EXTERNAL_INPUT: u32 = 0x00000001;
pub const XNN_VALUE_FLAG_EXTERNAL_OUTPUT: u32 = 0x00000002;
pub const XNN_EXTRA_BYTES: usize = 16;
pub const XNN_MAX_TENSOR_DIMS: usize = 6;
pub const XNN_FLAG_TRANSPOSE_WEIGHTS: u32 = 0x00000001;
pub const XNN_FLAG_TRANSPOSE_B: u32 = XNN_FLAG_TRANSPOSE_WEIGHTS;
pub const XNN_FLAG_TENSORFLOW_SAME_PADDING: u32 = 0x00000004;
pub const XNN_FLAG_KEEP_DIMS: u32 = 0x00000040;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum xnn_status {
    xnn_status_success = 0,
    xnn_status_uninitialized = 1,
    xnn_status_invalid_parameter = 2,
    xnn_status_invalid_state = 3,
    xnn_status_unsupported_parameter = 4,
    xnn_status_unsupported_hardware = 5,
    xnn_status_out_of_memory = 6,
    xnn_status_reallocation_required = 7,
    xnn_status_deprecated = 8,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum xnn_datatype {
    xnn_datatype_invalid = 0,
    xnn_datatype_fp32 = 1,
    xnn_datatype_fp16 = 2,
    xnn_datatype_qint8 = 3,
    xnn_datatype_quint8 = 4,
    xnn_datatype_qint32 = 5,
    xnn_datatype_qcint8 = 6,
    xnn_datatype_qcint32 = 7,
    xnn_datatype_qcint4 = 8,
    xnn_datatype_qdint8 = 9,
    xnn_datatype_qpint8 = 10,
    xnn_datatype_int32 = 11,
    xnn_datatype_qbint4 = 12,
    xnn_datatype_pfp32 = 13,
    xnn_datatype_bf16 = 14,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum xnn_unary_operator {
    xnn_unary_invalid = -1,
    xnn_unary_abs = 0,
    xnn_unary_approxgelu,
    xnn_unary_bankers_rounding,
    xnn_unary_ceiling,
    xnn_unary_clamp,
    xnn_unary_convert,
    xnn_unary_cosine,
    xnn_unary_elu,
    xnn_unary_exp,
    xnn_unary_floor,
    xnn_unary_gelu,
    xnn_unary_hardswish,
    xnn_unary_leaky_relu,
    xnn_unary_log,
    xnn_unary_negate,
    xnn_unary_reciprocal_square_root,
    xnn_unary_sigmoid,
    xnn_unary_sine,
    xnn_unary_square_root,
    xnn_unary_square,
    xnn_unary_tanh,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum xnn_binary_operator {
    xnn_binary_invalid = -1,
    xnn_binary_add = 0,
    xnn_binary_subtract,
    xnn_binary_multiply,
    xnn_binary_divide,
    xnn_binary_maximum,
    xnn_binary_minimum,
    xnn_binary_copysign,
    xnn_binary_squared_difference,
    xnn_binary_prelu,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union xnn_unary_params {
    pub clamp: xnn_unary_clamp_params,
    pub elu: xnn_unary_elu_params,
    pub leaky_relu: xnn_unary_leaky_relu_params,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_unary_clamp_params {
    pub min: f32,
    pub max: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_unary_elu_params {
    pub alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_unary_leaky_relu_params {
    pub negative_slope: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_binary_params {
    pub output_min: f64,
    pub output_max: f64,
}

#[repr(C)]
pub struct xnn_external_value {
    pub id: u32,
    pub data: *mut c_void,
}

unsafe extern "C" {
    pub fn xnn_initialize(allocator: *const c_void) -> xnn_status;
    pub fn xnn_deinitialize() -> xnn_status;

    pub fn xnn_create_subgraph(
        external_value_ids: u32,
        flags: u32,
        subgraph_out: *mut xnn_subgraph_t,
    ) -> xnn_status;

    pub fn xnn_delete_subgraph(subgraph: xnn_subgraph_t) -> xnn_status;

    pub fn xnn_define_tensor_value(
        subgraph: xnn_subgraph_t,
        datatype: xnn_datatype,
        num_dims: usize,
        dims: *const usize,
        data: *const c_void,
        external_id: u32,
        flags: u32,
        id_out: *mut u32,
    ) -> xnn_status;

    pub fn xnn_define_convolution_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        kernel_height: u32,
        kernel_width: u32,
        subsampling_height: u32,
        subsampling_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        groups: u32,
        group_input_channels: usize,
        group_output_channels: usize,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_depthwise_convolution_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        kernel_height: u32,
        kernel_width: u32,
        subsampling_height: u32,
        subsampling_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        depth_multiplier: u32,
        input_channels: usize,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_max_pooling_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        pooling_height: u32,
        pooling_width: u32,
        stride_height: u32,
        stride_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_global_average_pooling_2d(
        subgraph: xnn_subgraph_t,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_average_pooling_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        pooling_height: u32,
        pooling_width: u32,
        stride_height: u32,
        stride_width: u32,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_fully_connected(
        subgraph: xnn_subgraph_t,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_softmax(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_unary(
        subgraph: xnn_subgraph_t,
        op_type: xnn_unary_operator,
        params: *const xnn_unary_params,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_binary(
        subgraph: xnn_subgraph_t,
        op_type: xnn_binary_operator,
        params: *const xnn_binary_params,
        input1_id: u32,
        input2_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_static_reshape(
        subgraph: xnn_subgraph_t,
        num_dims: usize,
        new_shape: *const usize,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_static_transpose(
        subgraph: xnn_subgraph_t,
        num_dims: usize,
        perm: *const usize,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_concatenate(
        subgraph: xnn_subgraph_t,
        axis: usize,
        num_inputs: usize,
        inputs: *const u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_batch_matrix_multiply(
        subgraph: xnn_subgraph_t,
        input1_id: u32,
        input2_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_create_runtime(
        subgraph: xnn_subgraph_t,
        runtime_out: *mut xnn_runtime_t,
    ) -> xnn_status;

    pub fn xnn_create_runtime_v2(
        subgraph: xnn_subgraph_t,
        threadpool: pthreadpool_t,
        flags: u32,
        runtime_out: *mut xnn_runtime_t,
    ) -> xnn_status;

    pub fn xnn_reshape_runtime(runtime: xnn_runtime_t) -> xnn_status;

    pub fn xnn_setup_runtime_v2(
        runtime: xnn_runtime_t,
        num_external_values: usize,
        external_values: *const xnn_external_value,
    ) -> xnn_status;

    pub fn xnn_invoke_runtime(runtime: xnn_runtime_t) -> xnn_status;

    pub fn xnn_delete_runtime(runtime: xnn_runtime_t) -> xnn_status;
}

/// Ensure XNNPACK is initialized (idempotent).
pub fn ensure_init() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let status = unsafe { xnn_initialize(std::ptr::null()) };
        assert_eq!(
            status,
            xnn_status::xnn_status_success,
            "xnn_initialize failed: {status:?}"
        );
    });
}
