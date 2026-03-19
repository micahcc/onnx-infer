{
  lib,
  stdenv,
  rustPlatform,
  flakeInputs,
}:
let
	pname = "onnx-infer";
in
rustPlatform.buildRustPackage {
  inherit pname ;
  version = "0.0.0";

  src = "${flakeInputs.${pname}}";

  strictDeps = true;

  cargoLock.lockFile = "${flakeInputs.${pname}}/Cargo.lock";

  nativeBuildInputs = [ ];

  cargoBuildFlags = [ "--package=${pname}" ];
  cargoTestFlags = [ "--package=${pname}" ];
}
