import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from _deepposekit_gpu import bundled_cuda_installed, main, prepare_cuda


@unittest.skipUnless(sys.platform == "linux", "GPU launcher supports Linux and WSL2")
class GPULauncherTests(unittest.TestCase):
    def launch_environment(self, xla_flags, argv=None, *, automatic=False):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / 'Mødel env "quoted"'
            nvidia = site / "nvidia"
            compiler_bin = nvidia / "cuda_nvcc" / "bin"
            compiler_bin.mkdir(parents=True)
            (compiler_bin / "ptxas").touch()
            (nvidia / "cusolver" / "lib").mkdir(parents=True)
            original = {
                "LD_LIBRARY_PATH": "/opt/other/lib",
                "PATH": "/usr/bin",
                "CUDA_VISIBLE_DEVICES": "-1",
            }
            if xla_flags is not None:
                original["XLA_FLAGS"] = xla_flags
            with (
                patch.dict(os.environ, original, clear=True),
                patch("_deepposekit_gpu.sysconfig.get_path", return_value=str(site)),
                patch("_deepposekit_gpu.os.execve") as execute,
            ):
                command = prepare_cuda if automatic else main
                command(argv)
                if argv is not None:
                    self.assertEqual(execute.call_args.args[1], [sys.executable, *argv])
                environment = execute.call_args.args[2]
                if automatic:
                    execute.reset_mock()
                    with patch.dict(os.environ, environment, clear=True):
                        self.assertTrue(bundled_cuda_installed())
                        prepare_cuda(argv)
                    execute.assert_not_called()
            return environment, nvidia

    def test_automatic_cuda_setup_is_idempotent_and_preserves_device_masks(self):
        environment, _ = self.launch_environment(
            "--xla_gpu_autotune_level=0",
            ["-P", "-m", "deepposekit", "doctor"],
            automatic=True,
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(environment["XLA_FLAGS"], "--xla_gpu_autotune_level=0")

    def test_missing_cuda_wheels_skips_automatic_restart(self):
        with (
            tempfile.TemporaryDirectory() as site,
            patch("_deepposekit_gpu.sysconfig.get_path", return_value=site),
            patch("_deepposekit_gpu.os.execve") as execute,
        ):
            self.assertFalse(bundled_cuda_installed())
            prepare_cuda(["-P", "-m", "deepposekit", "doctor"])
        execute.assert_not_called()

    def test_embedded_launcher_accepts_an_argument_vector(self):
        self.launch_environment(None, ["-m", "downstream", "doctor"])

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
