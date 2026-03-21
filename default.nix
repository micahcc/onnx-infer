{
  pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/5e40dc287bf1e4e2c372bafb7cfeae95a7419ee2.tar.gz") { },
}:
let
  xnnpack = pkgs.callPackage ./nix/xnnpack.nix { };
  pname = "onnx-infer";
  src = pkgs.lib.cleanSourceWith {
    src = pkgs.lib.cleanSource ./.;
    filter = path: type:
      !(baseNameOf path == "target" && type == "directory")
      && !(baseNameOf path == "fixtures" && type == "directory");
  };
in
pkgs.rustPlatform.buildRustPackage {
  inherit pname src;
  version = "0.0.0";

  XNNPACK = "${xnnpack}";

  strictDeps = true;

  nativeBuildInputs = [
    pkgs.cargo
    pkgs.rustc
    pkgs.clippy
    (pkgs.rustfmt.override { asNightly = true; })
  ];

  cargoLock.lockFile = "${src}/Cargo.lock";

  cargoBuildFlags = [ "--package=${pname}" ];
  cargoTestFlags = [ "--package=${pname}" ];
}
