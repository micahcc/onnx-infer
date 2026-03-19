{
  lib,
  stdenv,
  rustPlatform,
  libclang,
  flakeInputs,
  xnnpack,
}:
let
  pname = "onnx-infer";
in
rustPlatform.buildRustPackage {
  inherit pname;
  version = "0.0.0";

  src = "${flakeInputs.${pname}}";

  strictDeps = true;

  cargoLock.lockFile = "${flakeInputs.${pname}}/Cargo.lock";

  nativeBuildInputs = [ libclang ];

  cargoBuildFlags = [ "--package=${pname}" ];
  cargoTestFlags = [ "--package=${pname}" ];

  XNNPACK = "${xnnpack}";
  LIBCLANG_PATH = "${libclang.lib}/lib";
}
