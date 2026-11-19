"""Configure Python for the CUDA libraries bundled by the GPU extra."""

import os
from pathlib import Path
import sys
import sysconfig


def _cuda_installation():
    if sys.platform != "linux":
        return None
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    compiler_bin = nvidia / "cuda_nvcc" / "bin"
    solver_lib = nvidia / "cusolver" / "lib"
    if solver_lib.is_dir() and (compiler_bin / "ptxas").is_file():
        return nvidia, compiler_bin
    return None


def bundled_cuda_installed():
    return _cuda_installation() is not None


def _prepend_paths(paths, inherited):
    prefix = [str(path) for path in paths]
    existing = inherited.split(os.pathsep) if inherited is not None else ()
    return os.pathsep.join(
        [*prefix, *(path for path in existing if path not in prefix)]
    )


def _cuda_environment():
    installation = _cuda_installation()
    if installation is None:
        return None
    nvidia, compiler_bin = installation
    env = os.environ.copy()
    library_dirs = sorted(path for path in nvidia.glob("*/lib") if path.is_dir())
    wsl_driver = Path("/usr/lib/wsl/lib")
    if wsl_driver.is_dir():
        library_dirs.insert(0, wsl_driver)
    env["LD_LIBRARY_PATH"] = _prepend_paths(
        library_dirs, env.get("LD_LIBRARY_PATH") or None
    )
    env["PATH"] = _prepend_paths([compiler_bin], env.get("PATH", os.defpath))
    return env


def prepare_cuda(argv):
    """Re-execute the CLI if its installed CUDA paths need to be activated."""
    env = _cuda_environment()
    if env is not None and env != os.environ:
        os.execve(sys.executable, [sys.executable, *argv], env)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    env = _cuda_environment()
    if env is None:
        raise SystemExit(
            "CUDA dependencies are missing. Install with the gpu extra on "
            "x86-64 Linux or WSL2 with an NVIDIA driver."
        )
    os.execve(sys.executable, [sys.executable, *argv], env)
