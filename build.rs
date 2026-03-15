use std::env;
use std::io::Result;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use tar::Archive;

const ONNX_MODELS: &str = "https://github.com/onnx/models/raw/c32b9776d06d2ebc7888d705e3a558f62b20e7a8/validated/vision/classification";

/// Download and extract a fixture tarball.
/// `dir_in_tarball` is the top-level directory inside the tarball (e.g. "model").
/// `fixture_name` is the name we want in our fixtures directory (e.g. "mnist-7").
fn fetch_fixture(
    url: &str,
    dir_in_tarball: &str,
    fixture_name: &str,
    fixtures_dir: &PathBuf,
) -> Result<()> {
    let target = fixtures_dir.join(fixture_name);
    if target.exists() {
        return Ok(());
    }

    let response =
        reqwest::blocking::get(url).unwrap_or_else(|e| panic!("Failed to download {url}: {e}"));
    if !response.status().is_success() {
        panic!("Failed to download {url}: HTTP {}", response.status());
    }

    let decoder = GzDecoder::new(response);
    let mut archive = Archive::new(decoder);
    archive.unpack(fixtures_dir)?;

    // Rename if the directory inside the tarball differs from fixture_name
    let extracted = fixtures_dir.join(dir_in_tarball);
    if extracted != target && extracted.exists() {
        std::fs::rename(&extracted, &target)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let file_descriptors =
        protox::compile(["proto/onnx.proto"], ["proto/"]).expect("protox compile");
    prost_build::Config::new()
        .compile_fds(file_descriptors)
        .expect("prost codegen");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fixtures_dir = out_dir.join("fixtures");
    std::fs::create_dir_all(&fixtures_dir)?;

    // MNIST models (tarball -> dir_in_tarball -> fixture_name)
    let mnist_fixtures = [
        ("mnist-1.tar.gz", "mnist", "mnist-1"),
        ("mnist-7.tar.gz", "model", "mnist-7"),
        ("mnist-8.tar.gz", "model", "mnist-8"),
        ("mnist-12.tar.gz", "mnist-12", "mnist-12"),
        ("mnist-12-int8.tar.gz", "mnist-12-int8", "mnist-12-int8"),
    ];
    for (tarball, dir_in_tarball, fixture_name) in mnist_fixtures {
        fetch_fixture(
            &format!("{ONNX_MODELS}/mnist/model/{tarball}"),
            dir_in_tarball,
            fixture_name,
            &fixtures_dir,
        )?;
    }

    // MobileNetV2 models
    let mobilenet_fixtures = [
        ("mobilenetv2-7.tar.gz", "mobilenetv2-7", "mobilenetv2-7"),
        ("mobilenetv2-12.tar.gz", "mobilenetv2-12", "mobilenetv2-12"),
        (
            "mobilenetv2-12-int8.tar.gz",
            "mobilenetv2-12-int8",
            "mobilenetv2-12-int8",
        ),
        (
            "mobilenetv2-12-qdq.tar.gz",
            "mobilenetv2-12-qdq",
            "mobilenetv2-12-qdq",
        ),
    ];
    for (tarball, dir_in_tarball, fixture_name) in mobilenet_fixtures {
        fetch_fixture(
            &format!("{ONNX_MODELS}/mobilenet/model/{tarball}"),
            dir_in_tarball,
            fixture_name,
            &fixtures_dir,
        )?;
    }

    println!(
        "cargo:rustc-env=FIXTURES_DIR={}",
        fixtures_dir.to_str().unwrap()
    );

    Ok(())
}
