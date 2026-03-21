{
  lib,
  stdenv,
  rustPlatform,
  flakeInputs,
  rustPackages,
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
    rustPlatform.rust.cargo
    rustPlatform.rust.rustc
    rustPackages.clippy
  ];

  cargoLock.lockFile = "${flakeInputs.${pname}}/Cargo.lock";

  cargoBuildFlags = [ "--package=${pname}" ];
  cargoTestFlags = [ "--package=${pname}" ];
}
