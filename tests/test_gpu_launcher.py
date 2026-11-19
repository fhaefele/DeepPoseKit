import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from _deepposekit_gpu import main


@unittest.skipUnless(sys.platform == "linux", "GPU launcher supports Linux and WSL2")
class GPULauncherTests(unittest.TestCase):
    def launch_flags(self, flags):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / r'Mødel env "quoted" \ slash'
            compiler = site / "nvidia" / "cuda_nvcc"
            (compiler / "bin").mkdir(parents=True)
            (compiler / "bin" / "ptxas").touch()
            (site / "nvidia" / "cusolver" / "lib").mkdir(parents=True)
            with (
                patch.dict(os.environ, {"XLA_FLAGS": flags}, clear=True),
                patch("_deepposekit_gpu.sysconfig.get_path", return_value=str(site)),
                patch("_deepposekit_gpu.os.execve") as execute,
            ):
                main()
            return execute.call_args.args[2]["XLA_FLAGS"], compiler

    def test_preserves_an_xla_flag_file(self):
        original = "/tmp/dpk xla flags.txt"
        actual, _ = self.launch_flags(original)
        self.assertEqual(actual, original)

    def test_preserves_an_explicit_cuda_directory(self):
        original = '  --xla_gpu_cuda_data_dir="/opt/other CUDA"'
        actual, _ = self.launch_flags(original)
        self.assertEqual(actual, original)

    def test_appends_a_quoted_literal_cuda_directory(self):
        actual, compiler = self.launch_flags("--xla_gpu_autotune_level=0")
        self.assertEqual(
            shlex.split(actual),
            ["--xla_gpu_autotune_level=0", f"--xla_gpu_cuda_data_dir={compiler}"],
        )

    def test_flag_name_inside_another_value_does_not_hide_default(self):
        original = '--xla_dump_to="/tmp/--xla_gpu_cuda_data_dir investigation"'
        actual, compiler = self.launch_flags(original)
        self.assertEqual(
            shlex.split(actual),
            [
                "--xla_dump_to=/tmp/--xla_gpu_cuda_data_dir investigation",
                f"--xla_gpu_cuda_data_dir={compiler}",
            ],
        )
