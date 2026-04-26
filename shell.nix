{
  # tag 25.11
  pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/25.11.tar.gz") { }
,
}:
let
  xnnpack = pkgs.callPackage ./nix/xnnpack.nix { };
  package = import ./default.nix { inherit pkgs; };
  python-packages =
    p: with p; [
      flake8
      onnx
      tabulate
      pyyaml
      numpy
      loguru
      tabulate
      ipdb
      opencv4
      black
      pandas
      plotly
      scipy
      pytest
      pyquaternion
      requests
    ];

  python = pkgs.python3.withPackages python-packages;
in
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.gdb
    pkgs.lldb
    python
  ];
  inputsFrom = [ package ];
  XNNPACK = "${xnnpack}";
  LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
}
