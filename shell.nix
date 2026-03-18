let
  pkgs = import
    (fetchTarball (
      "https://github.com/NixOS/nixpkgs/archive/f0549fcd9330025d103fce421c6e17a4d70ed4d4.tar.gz"
    ))
    { };

  nixpkgs = import <nixpkgs> { };

  # Use nightly Rust with rustfmt and other tools
  rustChannels = nixpkgs.latest.rustChannels;
  rustNightly = (rustChannels.nightly.rust.override {
    extensions = [
      "rust-src"
      "rustfmt-preview"
      "clippy-preview"
      "rust-analysis"
    ];
  });
  xnnpack = pkgs.callPackage ./nix/xnnpack.nix { };
in
pkgs.mkShell {
  XNNPACK_DIR = "${xnnpack}";
  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  # Set environment variables for OpenBLAS
  LD_LIBRARY_PATH = "${pkgs.openblas}/lib";

  buildInputs = [
    # Rust nightly toolchain
    rustNightly

    # BLAS for optimized matrix operations
    pkgs.blas
    pkgs.xnnpack

    # Development tools
    pkgs.llvmPackages.clang
    pkgs.pkg-config

    # Optional debugging tools
    pkgs.lldb
    pkgs.gdb
  ];
}
