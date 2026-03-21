{
  # tag 25.11
  pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/25.11.tar.gz") { },
}:
let
  xnnpack = pkgs.callPackage ./nix/xnnpack.nix { };
  package = import ./default.nix { inherit pkgs; };
in
pkgs.mkShell {
  nativeBuildInputs = [ ];
  inputsFrom = [ package ];
  XNNPACK = "${xnnpack}";
  LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
}
