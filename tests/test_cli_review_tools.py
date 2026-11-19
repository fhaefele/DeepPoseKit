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
        nested = self.succeeds("imgaug", "toml", "BlendAlpha")
        (entry,) = tomllib.loads(nested)["generator"]["augmenter"]
        self.assertEqual(entry["foreground"][0]["name"], "GaussianBlur")
        self.assertEqual(invoke("imgaug", "toml", "SomeOf")[0], 1)
        with self.assertRaisesRegex(ValueError, "axis must be"):
            build_augmenters(data, [{"name": "FlipAxis", "axis": 2, "p": 1.0}])

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
        hdf5_source = self.path / "annotations.hdf5"
        hdf5_source.write_bytes(self.annotations.read_bytes())
        output = self.path / "prelabeled.hdf5"
        self.succeeds(
            "dataset",
            "prelabel",
            hdf5_source,
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

    def test_append_materializes_external_skeleton_in_new_location(self):
        metadata = self.path / "skeleton.h5"
        with (
            h5py.File(self.annotations, "r+") as annotated,
            h5py.File(metadata, "w") as file,
        ):
            for key in ("skeleton", "skeleton_names"):
                annotated.copy(key, file)
                del annotated[key]
                annotated[key] = h5py.ExternalLink(metadata.name, f"/{key}")
            file["skeleton"].attrs["label"] = "external"
            file["other_experiment"] = [42]
            annotated["other"] = h5py.ExternalLink(metadata.name, "/other_experiment")
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        output = self.path / "nested" / "expanded.hdf5"
        output.parent.mkdir()
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
        with h5py.File(output, "r") as file:
            self.assertEqual(file["skeleton"].file.id, file.id)
            self.assertEqual(file["skeleton"].attrs["label"], "external")
            np.testing.assert_array_equal(file["other"][:], [42])
            self.assertEqual(
                file["skeleton_names"].asstr()[:].tolist(), ["left", "right"]
            )

    def test_annotation_init_rejects_a_later_uint16_batch(self):
        reader = self.path / "mixed.py"
        reader.write_text(
            "import numpy as np\n"
            "def create_reader(path, *, batch_size, argv):\n"
            "    yield np.zeros((1, 32, 32, 1), dtype=np.uint8)\n"
            "    yield np.full((1, 32, 32, 1), 500, dtype=np.uint16)\n"
        )
        output = self.path / "mixed.h5"
        code, _, error = invoke(
            "dataset",
            "init",
            output,
            "--video",
            self.video,
            "--skeleton",
            self.skeleton,
            "--custom-reader",
            reader,
        )
        self.assertEqual(code, 1)
        self.assertIn("dtype uint8", error)
        self.assertFalse(output.exists())

    def test_two_diverse_frames_can_come_from_two_distinct_clusters(self):
        frames = np.asarray([0] * 5 + [255] * 5, dtype=np.uint8)
        video = self.path / "black-and-white.avi"
        writer = VideoWriter(str(video), (32, 32), fps=5)
        try:
            writer.write_batch(
                np.broadcast_to(frames[:, None, None, None], (10, 32, 32, 3))
            )
        finally:
            writer.close()
        output = self.path / "two.h5"
        self.succeeds(
            "dataset",
            "init",
            output,
            "--video",
            video,
            "--skeleton",
            self.skeleton,
            "--sample",
            "diverse",
            "--count",
            2,
            "--seed",
            42,
            "--gray",
        )
        with h5py.File(output, "r") as file:
            np.testing.assert_array_equal(file["images"][:, 0, 0, 0], [0, 255])

    def test_trace_reads_skeletons_independently_of_the_hdf5_image_key(self):
        skeleton = self.path / "annotation-metadata.hdf5"
        skeleton.write_bytes(self.annotations.read_bytes())
        with h5py.File(skeleton, "r+") as file:
            file.move("images", "camera2")
        predictions = self.path / "poses.npy"
        np.save(predictions, np.ones((2, 2, 3)))
        output = self.path / "trace.png"
        self.succeeds(
            "predictions",
            "trace",
            predictions,
            "--skeleton",
            skeleton,
            "--keypoint",
            "right",
            "--output",
            output,
        )
        numbered = self.path / "numbered.csv"
        numbered.write_text("name,parent,swap\n010,,20\n20,010,010\n")
        self.succeeds(
            "predictions",
            "trace",
            predictions,
            "--skeleton",
            numbered,
            "--keypoint",
            "010",
            "--output",
            output,
            "--overwrite",
        )

    def test_preview_renders_bgr_images_in_rgb_order(self):
        from matplotlib import pyplot as plt
        from deepposekit.cli.augmentations import _draw_panel

        figure, axes = plt.subplots(1, 2)
        try:
            image = np.zeros((6, 6, 3), dtype=np.uint8)
            image[1, 2] = [0, 0, 255]
            points = np.empty((0, 2))
            _draw_panel(axes[0], image, points, (), "BGR", rgb=False)
            _draw_panel(axes[1], image, points, (), "RGB", rgb=True)
            np.testing.assert_array_equal(
                axes[0].images[0].get_array()[1, 2], [255, 0, 0]
            )
            np.testing.assert_array_equal(
                axes[1].images[0].get_array()[1, 2], [0, 0, 255]
            )
        finally:
            plt.close(figure)

    def test_full_mode_defaults_to_a_reviewed_frame(self):
        from unittest.mock import patch
        from deepposekit.cli.augmentations import _draw_panel

        with h5py.File(self.annotations, "r+") as file:
            file["annotated"][0] = False
        config = self.path / "full.toml"
        config.write_text(
            '[data]\nannotations = "annotations.h5"\nmode = "full"\n'
            '[generator]\naugmenter = ["Affine"]\n'
        )
        with patch(
            "deepposekit.cli.augmentations._draw_panel", wraps=_draw_panel
        ) as draw:
            self.succeeds(
                "imgaug",
                "preview",
                "--config",
                config,
                "--samples",
                1,
                "--output",
                self.path / "full.png",
            )
        np.testing.assert_array_equal(draw.call_args_list[0].args[1], self.images[1])

    def test_append_keeps_prior_output_if_fixed_arrays_have_file_references(self):
        with h5py.File(self.annotations, "r+") as file:
            originals = file["images"][:]
            del file["images"]
            file.create_dataset("images", data=originals)
            file.attrs["original_images"] = file["images"].ref
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        output = self.path / "expanded.h5"
        output.write_bytes(b"prior output")
        code, _, error = invoke(
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
        self.assertIn("object references", error)
        self.assertEqual(output.read_bytes(), b"prior output")
        with h5py.File(self.annotations, "r") as file:
            self.assertEqual(file[file.attrs["original_images"]].name, "/images")

    def test_append_rejects_external_skeleton_with_reference_attributes(self):
        metadata = self.path / "skeleton.h5"
        with (
            h5py.File(self.annotations, "r+") as annotated,
            h5py.File(metadata, "w") as file,
        ):
            annotated.copy("skeleton", file)
            annotated.copy("skeleton_names", file)
            file["skeleton"].attrs["names"] = file["skeleton_names"].ref
            del annotated["skeleton"]
            annotated["skeleton"] = h5py.ExternalLink(metadata.name, "/skeleton")
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        output = self.path / "expanded.h5"
        code, _, error = invoke(
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
        self.assertEqual(code, 1)
        self.assertIn("object references", error)
        self.assertFalse(output.exists())

    def test_annotation_tools_protect_linked_metadata_from_overwrite(self):
        metadata = self.path / "metadata.h5"
        with (
            h5py.File(self.annotations, "r+") as annotated,
            h5py.File(metadata, "w") as file,
        ):
            annotated.copy("skeleton", file)
            del annotated["skeleton"]
            annotated["skeleton"] = h5py.ExternalLink(metadata.name, "/skeleton")
            file["other_experiment"] = [42]
        before = metadata.read_bytes()
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        for arguments in (
            (
                "append",
                self.annotations,
                "--source",
                self.video,
                "--frames",
                candidates,
            ),
            ("prelabel", self.annotations, "--model", self.path / "unused.keras"),
        ):
            code, _, error = invoke(
                "dataset", *arguments, "--output", metadata, "--overwrite"
            )
            self.assertEqual(code, 1)
            self.assertIn("would replace an input", error)
            self.assertEqual(metadata.read_bytes(), before)

    def test_trace_distinguishes_numeric_indices_from_explicit_names(self):
        from unittest.mock import patch

        names = self.path / "numeric.csv"
        names.write_text("name,parent,swap\n1,,0\n0,1,1\n")
        predictions = self.path / "poses.npy"
        values = np.ones((2, 2, 3))
        values[:, :, 0] = [[10, 20], [11, 21]]
        np.save(predictions, values)
        for args, expected in (
            ([], [10, 11]),
            (["--keypoint", "0"], [10, 11]),
            (["--keypoint-name", "0"], [20, 21]),
        ):
            with patch("matplotlib.axes.Axes.plot", autospec=True) as plot:
                self.succeeds(
                    "predictions",
                    "trace",
                    predictions,
                    "--skeleton",
                    names,
                    "--output",
                    self.path / "trace.png",
                    "--overwrite",
                    *args,
                )
                np.testing.assert_array_equal(plot.call_args_list[0].args[2], expected)

    def test_virtual_skeleton_protects_backing_relative_to_its_owner(self):
        folder = self.path / "metadata"
        folder.mkdir()
        backing = folder / "graph.h5"
        with (
            h5py.File(self.annotations, "r+") as annotated,
            h5py.File(backing, "w") as file,
        ):
            file["graph"] = annotated["skeleton"][:]
            file["unrelated"] = [42]
            del annotated["skeleton"]
            annotated["skeleton"] = h5py.ExternalLink(
                "metadata/skeleton.h5", "/skeleton"
            )
        with h5py.File(folder / "skeleton.h5", "w") as file:
            layout = h5py.VirtualLayout(shape=(2, 2), dtype=np.int32)
            layout[:] = h5py.VirtualSource("graph.h5", "/graph", shape=(2, 2))
            file.create_virtual_dataset("skeleton", layout, fillvalue=-99)
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        prior = backing.read_bytes()
        code, _, error = invoke(
            "dataset",
            "append",
            self.annotations,
            "--source",
            self.video,
            "--frames",
            candidates,
            "--output",
            backing,
            "--overwrite",
            "--gray",
        )
        self.assertEqual(code, 1)
        self.assertIn("would replace an input", error)
        self.assertEqual(backing.read_bytes(), prior)

    def test_copy_through_symlink_preserves_relative_external_link(self):
        metrics = self.path / "metrics.h5"
        with h5py.File(metrics, "w") as file:
            file["readings"] = [21, 42]
        with h5py.File(self.annotations, "r+") as file:
            file["metrics"] = h5py.ExternalLink(metrics.name, "/readings")
        alias = self.path / "link" / "annotations.h5"
        alias.parent.mkdir()
        alias.symlink_to(self.annotations)
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        output = alias.parent / "expanded.h5"
        self.succeeds(
            "dataset",
            "append",
            alias,
            "--source",
            self.video,
            "--frames",
            candidates,
            "--output",
            output,
            "--gray",
        )
        with h5py.File(output, "r") as file:
            np.testing.assert_array_equal(file["metrics"][:], [21, 42])

    def test_relocation_rejects_relative_virtual_metadata_atomically(self):
        metrics = self.path / "metrics.h5"
        with h5py.File(metrics, "w") as file:
            file["readings"] = [21, 42]
        with h5py.File(self.annotations, "r+") as file:
            layout = h5py.VirtualLayout(shape=(2,), dtype=np.int32)
            layout[:] = h5py.VirtualSource(metrics.name, "/readings", shape=(2,))
            file.create_virtual_dataset("virtual_metrics", layout, fillvalue=-999)
        candidates = self.path / "one.csv"
        candidates.write_text("frame\n2\n")
        output = self.path / "elsewhere" / "expanded.h5"
        output.parent.mkdir()
        output.write_bytes(b"prior output")
        code, _, error = invoke(
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
        self.assertIn("relative HDF5 backing paths", " ".join(error.split()))
        self.assertEqual(output.read_bytes(), b"prior output")
        with h5py.File(self.annotations, "r") as file:
            np.testing.assert_array_equal(file["virtual_metrics"][:], [21, 42])
