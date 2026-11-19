from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

import cv2
import h5py
import numpy as np

from deepposekit.cli import main
from deepposekit.cli.augmentations import build_augmenters
from deepposekit.io import (
    DataGenerator,
    TrainingGenerator,
    VideoWriter,
    initialize_dataset,
)
from deepposekit.models import LEAP


def invoke(*args):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main([str(value) for value in args])
    return code, out.getvalue(), err.getvalue()


class ReviewTools(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.skeleton = self.path / "skeleton.csv"
        self.skeleton.write_text("name,parent,swap\nleft,,right\nright,left,left\n")
        self.images = np.zeros((8, 32, 32, 1), dtype=np.uint8)
        for index, image in enumerate(self.images):
            image[:, : 10 + index] = 10 * index
            image[2, 3, 0] = 255
        self.points = np.tile([[3.0, 2.0], [8.0, 7.0]], (len(self.images), 1, 1))
        self.annotations = self.path / "annotations.h5"
        initialize_dataset(
            str(self.annotations), self.images, str(self.skeleton), self.points
        )
        with h5py.File(self.annotations, "r+") as file:
            file["annotated"][0:2] = True
            file["annotated"][2, 0] = True
            file.attrs["lab"] = "fixture"
        self.video = self.path / "frames.avi"
        writer = VideoWriter(str(self.video), (32, 32), fps=5)
        try:
            writer.write_batch(np.repeat(self.images, 3, axis=-1))
        finally:
            writer.close()

    def succeeds(self, *args):
        code, out, err = invoke(*args)
        self.assertEqual(code, 0, err)
        return out

    def test_imgaug_toml_order_flip_and_seeded_preview_forms(self):
        names = self.succeeds("imgaug", "list").splitlines()
        self.assertIn("Affine", names)
        self.assertIn("FlipAxis", names)
        toml = self.succeeds("imgaug", "toml", "Affine", "FlipAxis", "GaussianBlur")
        self.assertEqual(
            [item["name"] for item in tomllib.loads(toml)["generator"]["augmenter"]],
            ["Affine", "FlipAxis", "GaussianBlur"],
        )
        data = DataGenerator(str(self.annotations))
        (flip,) = build_augmenters(data, [{"name": "FlipAxis", "axis": 1, "p": 1.0}])
        transformed, labels = flip(images=self.images[:1], keypoints=self.points[:1])
        np.testing.assert_array_equal(transformed[0], self.images[0, :, ::-1])
        np.testing.assert_array_equal(labels, [[[23, 7], [28, 2]]])
        entry = '{ name = "Multiply", mul = [0.8, 1.2] }'
        data_toml = f'[data]\nannotations = "annotations.h5"\n\n[generator]\naugmenter = [{entry}]\n'
        minimal, full = self.path / "augment.toml", self.path / "training.toml"
        minimal.write_text(data_toml)
        full.write_text(
            data_toml + '\n[model]\nname = "LEAP"\n\n[training]\nbatch_size = 1\n'
        )
        grids = []
        for name, settings in (
            ("minimal", ["--config", minimal]),
            ("full", ["--config", full]),
            ("cli", ["--annotations", self.annotations, "--augment", entry]),
            ("override", ["--config", full, "--augment", entry]),
        ):
            output = self.path / f"{name}.png"
            self.succeeds(
                "imgaug",
                "preview",
                *settings,
                "--frame",
                1,
                "--samples",
                3,
                "--seed",
                7,
                "--output",
                output,
            )
            self.assertGreater(output.stat().st_size, 100)
            grids.append(output.read_bytes())
        self.assertTrue(all(grid == grids[0] for grid in grids))
        self.assertEqual(invoke("imgaug", "toml", "NoSuchAugmenter")[0], 1)

    def test_saved_overlay_geometry_trace_and_failed_replacement(self):
        reader = self.path / "reader.py"
        reader.write_text(
            "import numpy as np\n"
            "from deepposekit.io import VideoReader\n"
            "def create_reader(path, *, batch_size, argv):\n"
            "    assert argv == ['--rotated']\n"
            "    reader = VideoReader(str(path), batch_size, gray=True)\n"
            "    try:\n"
            "        for batch in reader.iter_batches():\n"
            "            yield np.rot90(batch, axes=(1, 2))\n"
            "    finally:\n"
            "        reader.close()\n"
        )
        values = np.concatenate(
            (self.points, np.ones((len(self.images), 2, 1))), axis=-1
        )
        predictions = self.path / "poses.npy"
        np.save(predictions, values)
        trace = self.path / "trace.png"
        self.succeeds(
            "predictions",
            "trace",
            predictions,
            "--keypoint",
            "left",
            "--skeleton",
            self.skeleton,
            "--output",
            trace,
        )
        self.assertTrue(trace.read_bytes().startswith(b"\x89PNG"))
        self.assertIn(
            "Keypoint index",
            invoke(
                "predictions", "trace", predictions, "--keypoint", 2, "--output", trace
            )[2],
        )
        for kind, options, marker in (
            ("video", [], (2, 3)),
            (
                "rotated",
                ["--custom-reader", reader, "--custom-reader-args", "--rotated"],
                (28, 2),
            ),
        ):
            output = self.path / f"{kind}.avi"
            self.succeeds(
                "predictions",
                "view",
                predictions,
                "--source",
                self.video,
                "--stop",
                2,
                "--output",
                output,
                *options,
            )
            camera = cv2.VideoCapture(str(output))
            try:
                self.assertEqual(int(camera.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                ok, frame = camera.read()
                self.assertTrue(ok)
                self.assertGreater(frame[2, 3, 1], 230)
                if kind == "rotated":
                    np.testing.assert_array_equal(frame[marker], [255, 255, 255])
            finally:
                camera.release()
        np.save(predictions, values[:-1])
        previous = output.read_bytes()
        code, _, error = invoke(
            "predictions",
            "view",
            predictions,
            "--source",
            self.video,
            "--output",
            output,
            "--overwrite",
        )
        self.assertEqual(code, 1)
        self.assertIn("more than", error)
        self.assertEqual(output.read_bytes(), previous)

    def test_outlier_csv_appends_in_rank_order_to_fixed_legacy_arrays(self):
        values = np.zeros((len(self.images), 2, 3), dtype=float)
        values[..., 2] = 1
        values[2:, 0, 0] = 20
        values[2:, 0, 2] = 0.4
        values[5:, 1, 1] = 80
        values[5:, 1, 2] = 0.2
        predictions = self.path / "outliers.npy"
        np.save(predictions, values)
        candidates = self.path / "candidates.csv"
        self.succeeds(
            "predictions", "outliers", predictions, "--top", 3, "--output", candidates
        )
        with candidates.open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(row["frame"]) for row in rows], [5, 2])
        self.assertTrue(all(row["signals"] == "confidence_drop+jump" for row in rows))
        with h5py.File(self.annotations, "r+") as file:
            for name in ("images", "annotations", "annotated"):
                original = file[name][:]
                del file[name]
                file.create_dataset(name, data=original).attrs["unit"] = "original"
        source = self.annotations.read_bytes()
        output = self.path / "expanded.h5"
        self.succeeds(
            "dataset",
            "append",
            self.annotations,
            "--source",
            self.video,
            "--frames",
            candidates,
            "--output",
            output,
            "--gray",
        )
        self.assertEqual(self.annotations.read_bytes(), source)
        with (
            h5py.File(output, "r") as expanded,
            h5py.File(self.annotations, "r") as original,
        ):
            for name in (
                "images",
                "annotations",
                "annotated",
                "skeleton",
                "skeleton_names",
            ):
                np.testing.assert_array_equal(
                    expanded[name][: len(original[name])], original[name][:]
                )
            np.testing.assert_array_equal(expanded["images"][-2:], self.images[[5, 2]])
            np.testing.assert_array_equal(expanded["annotations"][-2:], -1)
            self.assertFalse(expanded["annotated"][-2:].any())
            self.assertEqual(expanded["images"].attrs["unit"], "original")
            self.assertEqual(expanded.attrs["lab"], "fixture")
        candidates.write_text("frame\n100\n")
        prior = output.read_bytes()
        code, _, err = invoke(
            "dataset",
            "append",
            self.annotations,
            "--source",
            self.video,
            "--frames",
            candidates,
            "--output",
            output,
            "--overwrite",
            "--gray",
        )
        self.assertEqual(code, 1)
        self.assertIn("outside", err)
        self.assertEqual(output.read_bytes(), prior)

    def test_prelabel_only_fills_unreviewed_keypoints(self):
        data = DataGenerator(str(self.annotations))
        generator = TrainingGenerator(data, downsample_factor=0, use_graph=False)
        model = LEAP(generator, filters=4)
        checkpoint = self.path / "model.keras"
        model.save(checkpoint)
        expected = model.predict_on_batch(self.images)[..., :2]
        output = self.path / "prelabeled.h5"
        self.succeeds(
            "dataset",
            "prelabel",
            self.annotations,
            "--model",
            checkpoint,
            "--output",
            output,
        )
        with (
            h5py.File(output, "r") as result,
            h5py.File(self.annotations, "r") as original,
        ):
            mask = original["annotated"][:]
            np.testing.assert_array_equal(result["annotated"][:], mask)
            np.testing.assert_array_equal(
                result["annotations"][:][mask], self.points[mask]
            )
            np.testing.assert_allclose(result["annotations"][:][~mask], expected[~mask])

    def test_diverse_sample_reproducibly_selects_actual_stride_frames(self):
        counts = []
        for run in (1, 2):
            output = self.path / f"diverse-{run}.h5"
            self.succeeds(
                "dataset",
                "init",
                output,
                "--video",
                self.video,
                "--gray",
                "--skeleton",
                self.skeleton,
                "--sample",
                "diverse",
                "--count",
                3,
                "--seed",
                42,
                "--frame-step",
                2,
                "--batch-size",
                3,
            )
            with h5py.File(output, "r") as file:
                chosen = [
                    next(
                        index
                        for index in range(0, len(self.images), 2)
                        if np.array_equal(frame, self.images[index])
                    )
                    for frame in file["images"][:]
                ]
                self.assertEqual(chosen, sorted(set(chosen)))
                self.assertEqual(len(chosen), 3)
                counts.append(chosen)
        self.assertEqual(*counts)
