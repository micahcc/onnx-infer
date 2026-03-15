use std::env;
use std::io::Result;
use std::path::PathBuf;
use std::process::Command;

const ONNX_MODELS: &str = "https://github.com/onnx/models/raw/c32b9776d06d2ebc7888d705e3a558f62b20e7a8/validated/vision/classification";

fn fetch_fixture(url: &str, name: &str, fixtures_dir: &PathBuf) -> Result<()> {
    let marker = fixtures_dir.join(name);
    if marker.exists() {
        return Ok(());
    }

    let tarball = fixtures_dir.join(format!("{name}.tar.gz"));

    let status = Command::new("curl")
        .args(["-sL", url, "-o"])
        .arg(&tarball)
        .status()?;
    if !status.success() {
        panic!("Failed to download {url}");
    }

    let status = Command::new("tar")
        .args(["xzf"])
        .arg(&tarball)
        .arg("-C")
        .arg(fixtures_dir)
        .status()?;
    if !status.success() {
        panic!("Failed to extract {name}");
    }

    std::fs::remove_file(&tarball)?;

    Ok(())
}

fn main() -> Result<()> {
    let file_descriptors = protox::compile(["proto/onnx.proto"], ["proto/"]).expect("protox compile");
    prost_build::Config::new()
        .compile_fds(file_descriptors)
        .expect("prost codegen");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fixtures_dir = out_dir.join("fixtures");
    std::fs::create_dir_all(&fixtures_dir)?;

    fetch_fixture(
        &format!("{ONNX_MODELS}/mnist/model/mnist-1.tar.gz"),
        "mnist",
        &fixtures_dir,
    )?;
    fetch_fixture(
        &format!("{ONNX_MODELS}/mnist/model/mnist-12.tar.gz"),
        "mnist-12",
        &fixtures_dir,
    )?;
    fetch_fixture(
        &format!("{ONNX_MODELS}/mobilenet/model/mobilenetv2-12.tar.gz"),
        "mobilenetv2-12",
        &fixtures_dir,
    )?;

    println!(
        "cargo:rustc-env=FIXTURES_DIR={}",
        fixtures_dir.to_str().unwrap()
    );

    Ok(())
}
