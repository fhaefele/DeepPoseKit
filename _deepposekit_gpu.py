"""Launch Python with the CUDA libraries bundled by the GPU extra."""

import os
from pathlib import Path
import re
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
    flag_text = xla_flags.lstrip()
    uses_flag_file = bool(flag_text) and not flag_text.startswith("-")
    if not uses_flag_file:
        # Consume quoted values so flag-like text inside an option is ignored.
        flags = re.finditer(
            '(?:^|\\s)(--[^\\s=]+)(?:=(?:"(?:[^"\\\\]|\\\\.)*"|\'[^\']*\'|\\S*))?',
            flag_text,
            re.DOTALL,
        )
        if not any(match[1] == "--xla_gpu_cuda_data_dir" for match in flags):
            cuda_dir = str(cuda_compiler).replace("\\", "\\\\").replace('"', '\\"')
            env["XLA_FLAGS"] = f'{xla_flags} --xla_gpu_cuda_data_dir="{cuda_dir}"'.strip()

    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)
