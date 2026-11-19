import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from _deepposekit_gpu import main


@unittest.skipUnless(sys.platform == "linux", "GPU launcher supports Linux and WSL2")
class GPULauncherTests(unittest.TestCase):
    def launch_environment(self, xla_flags):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / 'Mødel env "quoted"'
            nvidia = site / "nvidia"
            compiler_bin = nvidia / "cuda_nvcc" / "bin"
            compiler_bin.mkdir(parents=True)
            (compiler_bin / "ptxas").touch()
            (nvidia / "cusolver" / "lib").mkdir(parents=True)
            original = {"LD_LIBRARY_PATH": "/opt/other/lib", "PATH": "/usr/bin"}
            if xla_flags is not None:
                original["XLA_FLAGS"] = xla_flags
            with (
                patch.dict(os.environ, original, clear=True),
                patch("_deepposekit_gpu.sysconfig.get_path", return_value=str(site)),
                patch("_deepposekit_gpu.os.execve") as execute,
            ):
                main()
            return execute.call_args.args[2], nvidia

    def test_leaves_xla_flags_to_tensorflow(self):
        flags = (
            None,
            "/tmp/dpk xla flags.txt",
            '  --xla_gpu_cuda_data_dir="/opt/other CUDA"',
            '--xla_dump_to="/tmp/--xla_gpu_cuda_data_dir investigation"',
        )
        for original in flags:
            with self.subTest(xla_flags=original):
                actual, _ = self.launch_environment(original)
                self.assertEqual(actual.get("XLA_FLAGS"), original)

    def test_uses_cuda_wheel_paths_with_unicode_and_spaces(self):
        actual, nvidia = self.launch_environment(None)
        self.assertEqual(
            actual["PATH"].split(os.pathsep),
            [str(nvidia / "cuda_nvcc" / "bin"), "/usr/bin"],
        )
        libraries = actual["LD_LIBRARY_PATH"].split(os.pathsep)
        self.assertIn(str(nvidia / "cusolver" / "lib"), libraries)
        self.assertEqual(libraries[-1], "/opt/other/lib")
