//! Auto-generated FFI bindings for the XNNPACK subgraph API (via bindgen).

#![allow(non_camel_case_types, non_upper_case_globals, dead_code, clippy::all)]

include!(concat!(env!("OUT_DIR"), "/xnnpack_bindings.rs"));

/// Ensure XNNPACK is initialized (idempotent).
pub fn ensure_init() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let status = unsafe { xnn_initialize(std::ptr::null()) };
        assert_eq!(
            status,
            xnn_status_xnn_status_success,
            "xnn_initialize failed: {status:?}"
        );
    });
}
