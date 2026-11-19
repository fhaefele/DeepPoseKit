"""Launch Python with the CUDA libraries bundled by the GPU extra."""

import json
import os
from pathlib import Path
import sys
import sysconfig


def main():
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    cuda_compiler = nvidia / "cuda_nvcc"
    compiler_bin = cuda_compiler / "bin"
    solver_lib = nvidia / "cusolver" / "lib"
    if not solver_lib.is_dir() or not (compiler_bin / "ptxas").is_file():
        raise SystemExit(
            "CUDA dependencies are missing. Run 'uv sync --locked --extra gpu' "
            "on x86-64 Linux or WSL2 with an NVIDIA driver."
        )

    env = os.environ.copy()
    library_dirs = sorted(path for path in nvidia.glob("*/lib") if path.is_dir())
    wsl_driver = Path("/usr/lib/wsl/lib")
    if wsl_driver.is_dir():
        library_dirs.insert(0, wsl_driver)
    library_paths = [str(path) for path in library_dirs]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    env["PATH"] = os.pathsep.join((str(compiler_bin), env.get("PATH", os.defpath)))

    xla_flags = env.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir" not in xla_flags:
        cuda_dir = json.dumps(str(cuda_compiler))
        env["XLA_FLAGS"] = f"{xla_flags} --xla_gpu_cuda_data_dir={cuda_dir}".strip()

    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)
