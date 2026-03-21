{
  lib,
  stdenv,
  rustPlatform,
  flakeInputs,
  rustPackages,
  rustfmt,
  cargo,
  rustc,
  clippy,
}:
let
  pname = "onnx-infer";
in
rustPlatform.buildRustPackage {
  inherit pname;
  version = "0.0.0";

  src = "${flakeInputs.${pname}}";

  strictDeps = true;

  nativeBuildInputs = [
    cargo
    rustc
    clippy
    (rustfmt.override { asNightly = true; })
  ];

  cargoLock.lockFile = "${flakeInputs.${pname}}/Cargo.lock";

  cargoBuildFlags = [ "--package=${pname}" ];
  cargoTestFlags = [ "--package=${pname}" ];
}
