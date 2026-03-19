use std::io::Result;

fn main() -> Result<()> {
    #[cfg(feature = "accelerate")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    #[cfg(feature = "blas")]
    println!("cargo:rustc-link-lib=blas");

    #[cfg(feature = "xnnpack")]
    {
        let xnnpack_dir =
            std::env::var("XNNPACK").expect("XNNPACK must be set for xnnpack feature");
        println!("cargo:rustc-link-search=native={xnnpack_dir}/lib");
        println!("cargo:rustc-link-lib=static=XNNPACK");
        println!("cargo:rustc-link-lib=static=xnnpack-microkernels-prod");
        println!("cargo:rustc-link-lib=static=cpuinfo");
        println!("cargo:rustc-link-lib=static=pthreadpool");
        // kleidiai is needed on ARM only
        #[cfg(target_arch = "aarch64")]
        println!("cargo:rustc-link-lib=static=kleidiai");
        // Link C++ runtime for XNNPACK internals
        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=c++");
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=stdc++");

        let include_dir = format!("{xnnpack_dir}/include");
        println!("cargo:rerun-if-changed={include_dir}/xnnpack.h");

        let bindings = bindgen::Builder::default()
            .header(format!("{include_dir}/xnnpack.h"))
            .clang_arg(format!("-I{include_dir}"))
            .allowlist_function("xnn_.*")
            .allowlist_type("xnn_.*")
            .allowlist_var("XNN_.*")
            .allowlist_type("pthreadpool_t")
            .derive_debug(true)
            .derive_copy(true)
            .generate_comments(false)
            .generate()
            .expect("bindgen failed to generate XNNPACK bindings");

        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_dir.join("xnnpack_bindings.rs"))
            .expect("failed to write XNNPACK bindings");
    }

    let file_descriptors =
        protox::compile(["proto/onnx.proto"], ["proto/"]).expect("protox compile");

    // Write the file descriptor set for prost-reflect
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(
        out_dir.join("onnx_descriptor.bin"),
        prost::Message::encode_to_vec(&file_descriptors),
    )?;

    prost_build::Config::new()
        .compile_fds(file_descriptors)
        .expect("prost codegen");

    Ok(())
}
