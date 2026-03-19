{
  description = "ONNX Inference in Rust";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
      ...
    }:
    let
      supported-systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
    in
    flake-utils.lib.eachSystem supported-systems (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
	xnnpack = pkgs.callPackage ./nix/xnnpack.nix { };
      in
      rec {
        packages = {
          default = pkgs.callPackage ./package.nix {
	  	inherit xnnpack;
            flakeInputs.onnx-infer = pkgs.lib.cleanSource ./.;
          };
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [ ];
          inputsFrom = [ packages.default ];
	  XNNPACK="${xnnpack}";
	  LIBCLANG_PATH="${pkgs.libclang.lib}/lib";
        };
      }
    );
}
