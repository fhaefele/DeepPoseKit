from contextlib import ExitStack, chdir, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from rich.text import Text

from deepposekit.cli import console_main, main
from deepposekit.io import DataGenerator, VideoReader, VideoWriter
from deepposekit.models import load_model


def invoke(*args):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        status = main([str(value) for value in args])
    return status, out.getvalue(), err.getvalue()


class TerminalTests(unittest.TestCase):
    def test_executable_prepares_cuda_and_programmatic_calls_stay_in_process(self):
        with (
            patch.object(sys, "argv", ["deepposekit", "doctor"]),
            patch("_deepposekit_gpu.prepare_cuda") as setup,
            patch("deepposekit.cli.operations.run") as run,
        ):
            self.assertEqual(console_main(), 0)
            setup.assert_called_once_with(["-P", "-m", "deepposekit", "doctor"])
            self.assertEqual(run.call_args.args[0].command_name, "deepposekit")
            setup.reset_mock()
            self.assertEqual(main(["doctor"], cuda_module="deepposekit"), 0)
            setup.assert_not_called()

    def test_completion_skips_cuda_setup(self):
        with (
            patch.object(sys, "argv", ["deepposekit", "completion", "bash"]),
            patch("_deepposekit_gpu.prepare_cuda") as setup,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(console_main(), 0)
        setup.assert_not_called()

    def test_grouped_help_wraps_and_respects_color(self):
        class Terminal(StringIO):
            def isatty(self):
                return True

            def fileno(self):
                return 1

        for width in (40, 80):
            for flags in ([], ["predict"]):
                details = Terminal()
                with (
                    patch(
                        "os.get_terminal_size",
                        return_value=os.terminal_size((width, 24)),
                    ),
                    patch.dict(os.environ, {"TERM": "xterm"}, clear=True),
                    redirect_stdout(details),
                    self.assertRaises(SystemExit) as result,
                ):
                    main([*flags, "--help"])
                self.assertEqual(result.exception.code, 0)
                rendered = details.getvalue()
                self.assertIn("\x1b[", rendered)
                self.assertTrue(
                    all(
                        line.cell_len <= width
                        for line in Text.from_ansi(rendered).split()
                    )
                )
                if not flags:
                    self.assertIn("Model workflow", rendered)
                    self.assertIn("Data and inspection", rendered)

        for option, colored in (("auto", False), ("always", True), ("never", False)):
            output = StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as result:
                main(["--color", option, "--help"])
            self.assertEqual(result.exception.code, 0)
            self.assertEqual("\x1b[" in output.getvalue(), colored)

    def test_missing_reader_has_usage_error(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as result:
            main(
                [
                    "predict",
                    "model.keras",
                    "video.avi",
                    "result.npy",
                    "--custom-reader-args",
                ]
            )
        self.assertEqual(result.exception.code, 2)

    def test_output_keeps_distinct_case_on_case_sensitive_storage(self):
        from deepposekit.cli.operations import output_file

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.keras"
            checkpoint.touch()
            output = checkpoint.with_name("Model.keras")
            if output.exists():
                self.skipTest("Storage treats the paths as identical")
            self.assertEqual(output_file(output, ".keras", False, checkpoint), output)

    def test_predict_preserves_literal_paths_after_an_option(self):
        with patch("deepposekit.cli.operations.run") as run:
            status, _, errors = invoke(
                "predict",
                "model.keras",
                "clip.avi",
                "--gray",
                "--",
                "-output.npy",
            )
        self.assertEqual(status, 0, errors)
        args = run.call_args.args[0]
        self.assertEqual(args.paths, [Path("clip.avi"), Path("-output.npy")])
        self.assertTrue(args.gray)

    @unittest.skipUnless(
        sys.platform != "win32", "Shell completion harness uses Unix shells."
    )
    def test_completion_and_fast_module_entry(self):
        for shell in ("bash", "zsh"):
            status, script, stderr = invoke("completion", shell)
            self.assertEqual(status, 0, stderr)
            self.assertIn("deepposekit", script)
        entry = [sys.executable, "-m", "deepposekit"]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys\n"
                "try:\n"
                "    runpy.run_module('deepposekit', run_name='__main__')\n"
                "except SystemExit as error:\n"
                "    assert error.code == 0\n"
                "assert 'tensorflow' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Model workflow", result.stdout)
        with TemporaryDirectory() as directory:
            target = Path(directory) / "response"
            config = Path(directory) / "training.toml"
            config.touch()
            for line, prefix in (
                ("deepposekit tr", "train"),
                ("deepposekit dataset i", "init"),
                ("deepposekit predict --bat", "--batch-size"),
                ("deepposekit train tra", "training.toml"),
            ):
                target.touch()
                env = dict(
                    os.environ,
                    _ARGCOMPLETE="3",
                    _ARGCOMPLETE_STDOUT_FILENAME=str(target),
                    COMP_LINE="python -m " + line,
                    COMP_POINT=str(len(line) + 10),
                )
                subprocess.run(entry, cwd=directory, env=env, check=True)
                self.assertIn(prefix, target.read_text())


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.path = Path(scratch.name)
        cls.images = np.random.default_rng(37).integers(
            0, 256, (5, 64, 64, 1), dtype=np.uint8
        )
        cls.video = cls.path / "frames.avi"
        writer = VideoWriter(str(cls.video), (64, 64), codec="FFV1", fps=5)
        try:
            writer.write_batch(np.repeat(cls.images, 3, axis=-1))
        finally:
            writer.close()
        cls.skeleton = cls.path / "skeleton.csv"
        cls.skeleton.write_text("name,parent,swap\nhead,,tail\ntail,head,head\n")
        cls.annotations = cls.path / "annotations.h5"
        result, _, errors = invoke(
            "dataset",
            "init",
            cls.annotations,
            "--video",
            cls.video,
            "--skeleton",
            cls.skeleton,
            "--gray",
        )
        if result:
            raise AssertionError(errors)
        with h5py.File(cls.annotations, "r+") as file:
            file["annotations"][:] = np.tile([[10, 12], [36, 24]], (5, 1, 1))
            file["annotated"][:] = True
        cls.checkpoint = cls.path / "model.keras"
        cls.config = cls.path / "training.toml"
        cls.config.write_text(
            '[data]\nannotations = "annotations.h5"\n'
            "[generator]\nuse_graph = false\nsigma = 3\nshuffle = false\nrandom_seed = 5\n"
            'augmenter = [{name = "FlipAxis", axis = 1, p = 1.0}]\n'
            '[model]\nname = "StackedDenseNet"\nn_transitions = -3\npretrained_weights = "none"\ngrowth_rate = 8\nsubpixel = false\n'
            '[compile]\noptimizer = {name = "adam", learning_rate = 0.002}\nloss = "mse"\njit_compile = false\n'
            "[training]\nbatch_size = 2\nepochs = 1\nsteps_per_epoch = 1\nverbose = 2\n"
            'callbacks = ["TerminateOnNaN"]\n'
            '[output]\ncheckpoint = "model.keras"\n'
        )
        result, _, errors = invoke("train", cls.config)
        if result:
            raise AssertionError(errors)
        cls.model = load_model(cls.checkpoint, compile=False)

    def test_config_and_inspection(self):
        status, text, errors = invoke("--color", "always", "inspect", self.annotations)
        self.assertEqual(status, 0, errors)
        self.assertEqual(json.loads(text)["labeled_frames"], 5)
        status, text, errors = invoke("inspect", self.checkpoint)
        self.assertEqual(status, 0, errors)
        self.assertEqual(json.loads(text)["image_shape"], [64, 64, 1])
        metadata = self.model._checkpoint_metadata
        self.assertEqual(metadata["pose_model_config"]["config"]["growth_rate"], 8)
        training = metadata["train_generator_config"]["config"]
        self.assertEqual(training["sigma"], 3)
        self.assertEqual(training["skeleton_names"], ["head", "tail"])
        self.assertTrue(training["augmenter"])
        trained = load_model(self.checkpoint, compile=True)
        self.assertAlmostEqual(
            float(trained.train_model.optimizer.learning_rate), 0.002
        )
        self.assertEqual(int(trained.train_model.optimizer.iterations), 1)

    def test_predict_and_evaluate_include_short_final_batch(self):
        expected = self.model.predict(self.images, batch_size=2, verbose=0)
        output = self.path / "predictions.npy"
        command = (
            "predict",
            self.checkpoint,
            self.video,
            output,
            "--gray",
            "--batch-size",
            2,
        )
        status, _, errors = invoke(*command)
        self.assertEqual(status, 0, errors)
        np.testing.assert_allclose(np.load(output), expected, rtol=1e-6, atol=1e-6)
        status, _, errors = invoke(*command)
        self.assertEqual(status, 1)
        self.assertIn("--overwrite", errors)
        metrics = self.path / "metrics.json"
        status, _, errors = invoke(
            "evaluate",
            self.checkpoint,
            self.annotations,
            "--batch-size",
            2,
            "--output",
            metrics,
        )
        self.assertEqual(status, 0, errors)
        report = json.loads(metrics.read_text())
        self.assertEqual(report["frames"], 5)
        self.assertEqual(len(report["per_keypoint_mean_px"]), 2)

    def test_predict_explicit_files_with_one_model_load(self):
        other = self.path / "other recording.avi"
        other.write_bytes(self.video.read_bytes())
        output = self.path / "explicit-predictions"
        with patch(
            "deepposekit.cli.operations.load_model", return_value=self.model
        ) as load:
            status, stdout, errors = invoke(
                "predict",
                self.checkpoint,
                self.video,
                "--gray",
                other,
                "--batch-size",
                2,
                "--output-dir",
                output,
            )
        self.assertEqual(status, 0, errors)
        load.assert_called_once_with(self.checkpoint, compile=False)
        outputs = [output / "frames.npy", output / "other recording.npy"]
        self.assertEqual(stdout.splitlines(), list(map(str, outputs)))
        expected = self.model.predict(self.images, batch_size=2, verbose=0)
        for path in outputs:
            np.testing.assert_allclose(np.load(path), expected, rtol=1e-6, atol=1e-6)

    def test_predict_directory_filters_and_preserves_relative_paths(self):
        root = self.path / "recordings"
        nested = root / "session"
        nested.mkdir(parents=True)
        (root / "rec.avi").write_bytes(self.video.read_bytes())
        (nested / "rec.AVI").write_bytes(self.video.read_bytes())
        (root / "notes.txt").write_text("not a video")
        shallow = self.path / "shallow"
        recursive = root / "outputs"
        recursive.mkdir()
        (recursive / "previous.avi").write_text("ignored output subtree")
        with patch(
            "deepposekit.cli.operations.load_model", return_value=self.model
        ) as load:
            status, _, errors = invoke(
                "predict",
                self.checkpoint,
                root,
                "--gray",
                "--output-dir",
                shallow,
            )
            self.assertEqual(status, 0, errors)
            self.assertEqual(
                sorted(path.name for path in shallow.iterdir()), ["rec.npy"]
            )
            status, _, errors = invoke(
                "predict",
                self.checkpoint,
                root,
                "--gray",
                "--recursive",
                "--output-dir",
                recursive,
            )
            self.assertEqual(status, 0, errors)
            self.assertEqual(load.call_count, 2)
        self.assertEqual(
            {
                path.relative_to(recursive).as_posix()
                for path in recursive.rglob("*.npy")
            },
            {"rec.npy", "session/rec.npy"},
        )

    def test_batch_preflight_rejects_conflicts_before_writing(self):
        root = self.path / "conflicting-input"
        root.mkdir()
        source = root / "sample.avi"
        source.touch()
        duplicate = root / "Sample.mkv"
        duplicate.touch()
        output = self.path / "conflicting-output"
        with patch("deepposekit.cli.operations.load_model") as load:
            status, _, errors = invoke(
                "predict", self.checkpoint, root, "--output-dir", output, "--overwrite"
            )
            self.assertEqual(status, 1)
            self.assertIn("Output collision", errors)
            self.assertFalse(output.exists())

            duplicate.unlink()
            nested = root / "sample.npy"
            nested.mkdir()
            (nested / "recording.avi").touch()
            status, _, errors = invoke(
                "predict", self.checkpoint, root, "--recursive", "--output-dir", output
            )
            self.assertEqual(status, 1)
            self.assertIn("needed as a directory", errors)
            self.assertFalse(output.exists())

            output.mkdir()
            existing = output / "sample.npy"
            existing.write_bytes(b"existing prediction")
            status, _, errors = invoke(
                "predict", self.checkpoint, root, "--output-dir", output
            )
            self.assertEqual(status, 1)
            self.assertIn("--overwrite", errors)
            self.assertEqual(existing.read_bytes(), b"existing prediction")
            existing.unlink()
            os.link(source, existing)
            status, _, errors = invoke(
                "predict", self.checkpoint, root, "--output-dir", output, "--overwrite"
            )
            self.assertEqual(status, 1)
            self.assertIn("would replace an input", errors)
            self.assertTrue(existing.samefile(source))
            load.assert_not_called()

    def test_recursive_legacy_reader_ignores_colocated_bytecode(self):
        root = self.path / "colocated-reader"
        cache = root / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "reader.cpython-312.pyc").write_bytes(b"stale bytecode")
        recording = root / "a.npy"
        np.save(recording, self.images)
        fixture = root / "reader.py"
        fixture.write_text(
            "import numpy as np\n"
            "def create_reader(path, *, batch_size, argv):\n"
            "    assert create_reader.__code__.co_filename == __file__\n"
            "    yield np.load(path)\n"
        )
        with patch("deepposekit.cli.operations.load_model") as load:
            status, _, errors = invoke(
                "predict",
                self.checkpoint,
                root,
                "--output-dir",
                root,
                "--custom-reader",
                fixture,
            )
        self.assertEqual(status, 1)
        self.assertIn("separate --output-dir", errors)
        load.assert_not_called()
        output = root / "predictions"
        with (
            chdir(root),
            patch("deepposekit.cli.operations.load_model", return_value=self.model),
        ):
            status, _, errors = invoke(
                "predict",
                self.checkpoint,
                root,
                "--recursive",
                "--output-dir",
                output,
                "--custom-reader",
                fixture.name,
            )
        self.assertEqual(status, 0, errors)
        self.assertEqual(list(output.rglob("*.npy")), [output / "a.npy"])
        self.assertEqual(
            [path.name for path in cache.iterdir()], ["reader.cpython-312.pyc"]
        )

    def test_custom_reader_batches_keep_arguments_and_completed_outputs(self):
        root = self.path / "reader-input"
        root.mkdir()
        for name in ("a", "b"):
            np.save(root / f"{name}.npy", self.images)
        (root / "sidecar.txt").write_text("ignore")
        fixture = self.path / "batch_reader.py"
        fixture.write_text(
            "import numpy as np\n"
            'INPUT_EXTENSIONS = (".npy",)\n'
            "def create_reader(path, *, batch_size, argv):\n"
            '    assert argv == ["--", "--overwrite"]\n'
            "    argv.pop()\n"
            "    try:\n"
            "        yield np.load(path)[:batch_size]\n"
            '        if path.stem == "b":\n'
            '            raise ValueError("unreadable second batch")\n'
            "        yield np.load(path)[batch_size:]\n"
            "    finally:\n"
            '        path.with_suffix(".closed").touch()\n'
        )
        output = self.path / "reader-output"
        output.mkdir()
        prior = output / "b.npy"
        prior.write_bytes(b"prior prediction")
        with patch(
            "deepposekit.cli.operations.load_model", return_value=self.model
        ) as load:
            status, stdout, errors = invoke(
                "predict",
                self.checkpoint,
                root,
                "--output-dir",
                output,
                "--batch-size",
                3,
                "--custom-reader",
                fixture,
                "--overwrite",
                "--custom-reader-args",
                "--",
                "--overwrite",
            )
        self.assertEqual(status, 1)
        self.assertIn("b.npy", errors)
        self.assertIn("unreadable second batch", errors)
        self.assertEqual(stdout.splitlines(), [str(output / "a.npy")])
        self.assertEqual(np.load(output / "a.npy").shape, (5, 2, 3))
        self.assertEqual(prior.read_bytes(), b"prior prediction")
        self.assertTrue(all((root / f"{name}.closed").is_file() for name in ("a", "b")))
        load.assert_called_once()

    def test_custom_reader_arguments_and_cleanup_after_failure(self):
        data = self.path / "input.npy"
        np.save(data, self.images)
        fixture = self.path / "reader.py"
        fixture.write_text(
            "import argparse\nfrom pathlib import Path\nimport numpy as np\n"
            "def create_reader(path, *, batch_size, argv):\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--gray', action='store_true', required=True)\n"
            "    parser.add_argument('--camera', required=True)\n"
            "    parser.add_argument('selection', nargs='?')\n"
            "    options = parser.parse_args(argv)\n"
            "    if options.camera == 'Camera 1':\n"
            "        assert options.selection == '--help'\n"
            "    assert options.camera in ('Camera 1', 'bad shape')\n"
            "    frames = np.load(path)\n"
            "    def batches():\n"
            "        try:\n"
            "            for index in range(0, len(frames), batch_size):\n"
            "                batch = frames[index:index + batch_size]\n"
            "                yield batch if not index or options.camera == 'Camera 1' else batch[:, :1]\n"
            "        finally:\n"
            "            Path(path).with_suffix('.closed').touch()\n"
            "    return batches()\n"
        )
        output = self.path / "custom.npy"
        closed = data.with_suffix(".closed")
        args = (
            "predict",
            self.checkpoint,
            data,
            output,
            "--batch-size",
            2,
            "--custom-reader",
            fixture,
        )
        status, _, errors = invoke(
            *args,
            "--custom-reader-args",
            "--camera",
            "Camera 1",
            "--gray",
            "--",
            "--help",
        )
        self.assertEqual(status, 0, errors)
        self.assertTrue(closed.is_file())
        prior = output.read_bytes()
        np.testing.assert_allclose(
            np.load(output),
            self.model.predict(self.images, batch_size=2, verbose=0),
            rtol=1e-6,
            atol=1e-6,
        )
        closed.unlink()
        status, _, errors = invoke(
            *args,
            "--overwrite",
            "--custom-reader-args",
            "--camera",
            "bad shape",
            "--gray",
        )
        self.assertEqual(status, 1)
        self.assertIn("Batch 2", errors)
        self.assertTrue(closed.is_file())
        self.assertEqual(output.read_bytes(), prior)

    def test_fine_tune_with_existing_weights_and_reject_wrong_skeleton(self):
        config = self.path / "fine-tune.toml"
        config.write_text(
            self.config.read_text().replace('"model.keras"', '"fine-tuned.keras"')
        )
        status, _, errors = invoke("train", config, "--init-from", self.checkpoint)
        self.assertEqual(status, 0, errors)
        self.assertTrue((self.path / "fine-tuned.keras").is_file())
        alternate = self.path / "other.h5"
        alternate.write_bytes(self.annotations.read_bytes())
        with h5py.File(alternate, "r+") as file:
            file["skeleton_names"][:] = [b"tail", b"head"]
        config.write_text(config.read_text().replace('"annotations.h5"', '"other.h5"'))
        status, _, errors = invoke(
            "train", config, "--init-from", self.checkpoint, "--overwrite"
        )
        self.assertEqual(status, 1)
        self.assertIn("skeleton_names", errors)

    def test_invalid_training_geometry_leaves_existing_output(self):
        config = self.path / "invalid.toml"
        checksum = self.checkpoint.read_bytes()
        content = self.config.read_text()
        for original, invalid, message in (
            ("subpixel = false", "subpixel = 0", "[model].subpixel"),
            ("sigma = 3", "sigma = -1", "[generator].sigma"),
        ):
            with self.subTest(setting=invalid):
                config.write_text(content.replace(original, invalid))
                status, _, errors = invoke(
                    "--color", "never", "train", config, "--overwrite"
                )
                self.assertEqual(status, 1)
                self.assertIn(message, errors)
                self.assertEqual(checksum, self.checkpoint.read_bytes())

    def test_callback_template_cannot_overwrite_inputs(self):
        data = self.path / "1.h5"
        data.write_bytes(self.annotations.read_bytes())
        prior = data.read_bytes()
        config = self.path / "callback.toml"
        content = self.config.read_text()
        content = content.replace("annotations.h5", "1.h5").replace(
            "model.keras", "callback.keras"
        )
        template = '{name="ModelCheckpoint", filepath="{epoch}.h5", monitor="loss", save_best_only=false}'
        config.write_text(content.replace('"TerminateOnNaN"', template))
        status, _, errors = invoke("train", config, "--overwrite")
        self.assertEqual(status, 1)
        self.assertIn("Callback path conflicts", errors)
        self.assertEqual(data.read_bytes(), prior)
        self.assertFalse((self.path / "callback.keras").is_file())

    def test_checkpoint_template_and_backup_keep_existing_files(self):
        runs = self.path / "runs"
        first = runs / "1" / "checkpoint.keras"
        first.parent.mkdir(parents=True)
        first.write_bytes(b"existing output")
        shared = self.path / "shared"
        shared.mkdir()
        unrelated = shared / "keep"
        unrelated.write_text("unrelated contents")
        inactive = shared / "latest.weights.h5.bkp"
        inactive.write_bytes(b"previous generation")
        config = self.path / "backup.toml"
        content = self.config.read_text().replace("model.keras", "backup.keras")
        checkpoint = (
            '{name="ModelCheckpoint", filepath="runs/{epoch}/checkpoint.keras", '
            'monitor="loss", save_best_only=false}'
        )
        backup = '{name="BackupAndRestore", backup_dir="shared"}'
        config.write_text(content.replace('"TerminateOnNaN"', checkpoint))
        status, _, errors = invoke("train", config)
        self.assertEqual(status, 1)
        self.assertEqual(first.read_bytes(), b"existing output")
        self.assertIn("--overwrite", errors)
        self.assertTrue(
            any(line.startswith("deepposekit: error:") for line in errors.splitlines())
        )
        config.write_text(
            content.replace('"TerminateOnNaN"', f"{checkpoint}, {backup}")
        )
        status, _, errors = invoke("train", config, "--overwrite")
        self.assertEqual(status, 0, errors)
        load_model(first, compile=False)
        self.assertEqual(set(shared.iterdir()), {unrelated, inactive})
        self.assertEqual(inactive.read_bytes(), b"previous generation")
        self.assertEqual(unrelated.read_text(), "unrelated contents")

    def test_dataset_init_keeps_full_names_and_streams_frame_sampling(self):
        path = self.path / "many.avi"
        frames = np.tile(self.images, (4, 1, 1, 1))
        writer = VideoWriter(str(path), (64, 64), codec="FFV1", fps=5)
        try:
            writer.write_batch(np.repeat(frames, 3, axis=-1))
        finally:
            writer.close()
        skeleton = self.path / "long-name.csv"
        skeleton.write_text(
            "name,parent,swap\nhead,,\nleft_antenna_base,head,\nleft_antenna_tip,head,\n"
        )
        destination = self.path / "long-name.h5"
        status, _, errors = invoke(
            "dataset",
            "init",
            destination,
            "--video",
            path,
            "--skeleton",
            skeleton,
            "--frame-step",
            3,
            "--gray",
        )
        self.assertEqual(status, 0, errors)
        data = DataGenerator(str(destination), mode="full")
        self.assertEqual(
            data.skeleton_names, ["head", "left_antenna_base", "left_antenna_tip"]
        )
        np.testing.assert_array_equal(
            data.get_images(np.arange(len(data))), frames[::3]
        )
        np.testing.assert_array_equal(data.get_keypoints(np.arange(len(data))), -1)
        with h5py.File(destination, "r") as file:
            self.assertFalse(file["annotated"][:].any())

        complete = path.read_bytes()
        with patch(
            "deepposekit.cli.operations.VideoReader", wraps=VideoReader
        ) as wrapper:

            def overestimated_reader(*args, **kwargs):
                reader = VideoReader(*args, **kwargs)
                reader.n_frames += 10
                return reader

            wrapper.side_effect = overestimated_reader
            status, _, errors = invoke(
                "dataset",
                "init",
                destination,
                "--video",
                path,
                "--skeleton",
                skeleton,
                "--frame-step",
                3,
                "--gray",
                "--overwrite",
            )
            self.assertEqual(status, 0, errors)
        # Keep the AVI header and a few frames, but cut the tail before EOF.
        path.write_bytes(complete[: len(complete) // 2])
        prior = destination.read_bytes()
        status, _, errors = invoke(
            "dataset",
            "init",
            destination,
            "--video",
            path,
            "--skeleton",
            skeleton,
            "--gray",
            "--overwrite",
        )
        self.assertEqual(status, 1)
        self.assertIn("reported frames", errors)
        self.assertEqual(destination.read_bytes(), prior)

    def test_static_callback_paths_are_reserved_before_keras_opens_them(self):
        from deepposekit.cli.training import _callback_list, read_training_config

        first = self.path / "callbacks-first.toml"
        second = self.path / "callbacks-second.toml"
        content = self.config.read_text().replace(
            '"TerminateOnNaN"', '{name="CSVLogger", filename="shared-metrics.csv"}'
        )
        first.write_text(content)
        second.write_text(content.replace("model.keras", "second.keras"))
        reserved = self.path / "shared-metrics.csv"
        with ExitStack() as resources:
            _callback_list(read_training_config(first), None, False, resources)
            self.assertTrue(reserved.exists())
            with ExitStack() as other, self.assertRaises(FileExistsError):
                _callback_list(read_training_config(second), None, False, other)
        self.assertFalse(reserved.exists())

    def test_backup_directory_stays_reserved_through_final_checkpoint_publication(self):
        from deepposekit.cli.training import _callback_list, read_training_config

        first = self.path / "backup-first.toml"
        second = self.path / "backup-second.toml"
        content = self.config.read_text().replace(
            '"TerminateOnNaN"',
            '{name="BackupAndRestore", backup_dir="reserved-backup"}',
        )
        first.write_text(content)
        second.write_text(content.replace("model.keras", "second.keras"))
        reserved = self.path / "reserved-backup"
        with ExitStack() as resources:
            (callback,) = _callback_list(
                read_training_config(first), None, False, resources
            )
            callback.on_train_end()
            self.assertTrue(reserved.is_dir())
            with ExitStack() as other, self.assertRaises(FileExistsError):
                _callback_list(read_training_config(second), None, False, other)
        self.assertFalse(reserved.exists())

    def test_backup_remains_when_final_checkpoint_publication_fails(self):
        from deepposekit.models.engine import BaseModel

        config = self.path / "recovery.toml"
        content = self.config.read_text().replace("model.keras", "recovery.keras")
        config.write_text(
            content.replace(
                '"TerminateOnNaN"', '{name="BackupAndRestore", backup_dir="recovery"}'
            )
        )
        with patch.object(
            BaseModel, "save", side_effect=FileExistsError("concurrent output")
        ):
            status, _, errors = invoke("train", config)
        self.assertEqual(status, 1)
        self.assertIn("concurrent output", errors)
        self.assertTrue((self.path / "recovery" / "latest.weights.h5").is_file())
        self.assertTrue((self.path / "recovery" / "training_metadata.json").is_file())

    def test_fine_tune_reuses_normalized_leap_constructor_options(self):
        config = self.path / "leap.toml"
        content = (
            '[data]\nannotations = "annotations.h5"\n'
            "[generator]\ndownsample_factor = 0\nuse_graph = false\n"
            '[model]\nname = "LEAP"\nfilters = 4\nbatchnorm = true\nuse_bias = true\n'
            "[training]\nbatch_size = 2\nsteps_per_epoch = 1\nverbose = 2\n"
            '[output]\ncheckpoint = "leap.keras"\n'
        )
        config.write_text(content)
        status, _, errors = invoke("train", config)
        self.assertEqual(status, 0, errors)
        checkpoint = self.path / "leap.keras"
        config.write_text(content.replace("leap.keras", "leap-fine-tuned.keras"))
        status, _, errors = invoke("train", config, "--init-from", checkpoint)
        self.assertEqual(status, 0, errors)
        self.assertTrue((self.path / "leap-fine-tuned.keras").is_file())
