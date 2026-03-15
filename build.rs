use std::io::Result;

fn main() -> Result<()> {
    let file_descriptors =
        protox::compile(["proto/onnx.proto"], ["proto/"]).expect("protox compile");
    prost_build::Config::new()
        .compile_fds(file_descriptors)
        .expect("prost codegen");

    Ok(())
}
