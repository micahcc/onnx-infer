use std::io::Result;

fn main() -> Result<()> {
    #[cfg(feature = "accelerate")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    #[cfg(feature = "blas")]
    println!("cargo:rustc-link-lib=blas");

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
