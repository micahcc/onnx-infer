use std::env;
use std::io::Result;
use std::path::PathBuf;
use std::process::Command;

const MNIST_URL: &str = "https://github.com/onnx/models/raw/c32b9776d06d2ebc7888d705e3a558f62b20e7a8/validated/vision/classification/mnist/model/mnist-1.tar.gz";

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
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"])?;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fixtures_dir = out_dir.join("fixtures");
    std::fs::create_dir_all(&fixtures_dir)?;

    fetch_fixture(MNIST_URL, "mnist", &fixtures_dir)?;

    println!(
        "cargo:rustc-env=FIXTURES_DIR={}",
        fixtures_dir.to_str().unwrap()
    );

    Ok(())
}
