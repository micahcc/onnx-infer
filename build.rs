use std::io::Result;

fn main() -> Result<()> {
    #[cfg(feature = "accelerate")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    #[cfg(feature = "blas")]
    println!("cargo:rustc-link-lib=blas");

    #[cfg(feature = "xnnpack")]
    {
        let xnnpack_dir =
            std::env::var("XNNPACK_DIR").expect("XNNPACK_DIR must be set for xnnpack feature");
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
