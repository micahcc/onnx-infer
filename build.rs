use std::env;
use std::io::Result;
use std::path::PathBuf;
use std::process::Command;

const MNIST_BASE: &str = "https://github.com/onnx/models/raw/c32b9776d06d2ebc7888d705e3a558f62b20e7a8/validated/vision/classification/mnist/model";

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

fn find_protoc() -> Option<PathBuf> {
    if let Ok(p) = env::var("PROTOC") {
        return Some(PathBuf::from(p));
    }
    let output = Command::new("which")
        .arg("protoc")
        .output()
        .ok()?;
    if output.status.success() {
        let path = String::from_utf8(output.stdout).ok()?.trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    None
}

fn main() -> Result<()> {
    let mut config = prost_build::Config::new();
    if let Some(protoc) = find_protoc() {
        config.protoc_executable(protoc);
    }
    config.compile_protos(&["proto/onnx.proto"], &["proto/"])?;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fixtures_dir = out_dir.join("fixtures");
    std::fs::create_dir_all(&fixtures_dir)?;

    fetch_fixture(&format!("{MNIST_BASE}/mnist-1.tar.gz"), "mnist", &fixtures_dir)?;
    fetch_fixture(&format!("{MNIST_BASE}/mnist-12.tar.gz"), "mnist-12", &fixtures_dir)?;

    println!(
        "cargo:rustc-env=FIXTURES_DIR={}",
        fixtures_dir.to_str().unwrap()
    );

    Ok(())
}
