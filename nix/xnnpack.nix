{
  clangStdenv,
  fetchFromGitHub,
  cmake,
  python3,
}:
let
  cpuinfo_src = fetchFromGitHub {
    owner = "pytorch";
    repo = "cpuinfo";
    rev = "731932997e55e84900e6f95aefb6edd167b0edcc";
    sha256 = "sha256-BfndQo6FTnCHDhrV5qW6rmej9/ImEpIqrWNUXCDXSao=";
  };
  fxdiv_src = fetchFromGitHub {
    owner = "Maratyszcza";
    repo = "FXdiv";
    rev = "63058eff77e11aa15bf531df5dd34395ec3017c8";
    sha256 = "sha256-LjX5kivfHbqCIA5pF9qUvswG1gjOFo3CMpX0VR+Cn38=";
  };
  pthreadpool_src = fetchFromGitHub {
    owner = "google";
    repo = "pthreadpool";
    rev = "15a6644ba1c45f1acc16ac1e883efc3e56c6bed2";
    sha256 = "sha256-YJD9n8cxoqTrTQJHKuZFW2qnlNcNmh5rQlBpnlF46to=";
  };
  kleidiai_src = fetchFromGitHub {
    owner = "ARM-software";
    repo = "kleidiai";
    rev = "1ae1e7c5882e1dbcb3992890c1af0538629641df";
    sha256 = "sha256-4beu5OaN5//r5Hkq+6Yu37S3BjEftOA+tTpuMRphIk0=";
  };
  gbenchmark_src = fetchFromGitHub {
    owner = "google";
    repo = "benchmark";
    rev = "42786a7876120b03e2ecff269695729337d243d9";
    sha256 = "sha256-Oa3UX207eeWhLzrBZSzyyehVmso6e8ned9XPo0pExCU=";
  };
  gtest_src = fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    rev = "a8380b01a25ebdb130097b607b3afed2a58d04ba";
    sha256 = "sha256-ftKnwcFvnMvLquNJGVRRgb2mbWokl82qQ08E9PpgNXM=";
  };
in

clangStdenv.mkDerivation {
  name = "XNNPACK";
  src = fetchFromGitHub {
    owner = "google";
    repo = "XNNPACK";
    rev = "aa4235ff811a5ad7907fb060593ec88dc6d2428d";
    sha256 = "sha256-hbL7U3IWn5J7CHWyKY8+1LHSAXabRYYDk3of3KDbe9E=";
  };
  nativeBuildInputs = [
    cmake
    python3
  ];
  cmakeFlags = [
    "-DCPUINFO_SOURCE_DIR=${cpuinfo_src}"
    "-DFXDIV_SOURCE_DIR=${fxdiv_src}"
    "-DPTHREADPOOL_SOURCE_DIR=${pthreadpool_src}"
    "-DKLEIDIAI_SOURCE_DIR=${kleidiai_src}"
    "-DGOOGLEBENCHMARK_SOURCE_DIR=${gbenchmark_src}"
    "-DGOOGLETEST_SOURCE_DIR=${gtest_src}"
    "-DXNNPACK_ENABLE_KLEIDIAI=ON"
  ];
}
