import gc
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
import torch

from deepposekit import FlipAxis
from deepposekit.io import (
    DataGenerator,
    TrainingGenerator,
    VideoReader,
    VideoWriter,
    initialize_dataset,
)
from deepposekit.models import StackedDenseNet, load_model
from deepposekit.models.backend.backend import (
    depth_to_space,
    resize_images,
    space_to_depth,
)
from deepposekit.models.backend.utils import fftshift1d
from deepposekit.utils.keypoints import graph_to_edges

import keras


def write_lossless_video(path, frames):
    height, width = frames.shape[1:3]
    writer = VideoWriter(path, (width, height), codec="FFV1", fps=5)
    try:
        writer.write_batch(frames)
    finally:
        writer.close()


class StackedDenseNetWorkflowTests(unittest.TestCase):
    def test_subpixel_fftshift_for_odd_and_even_lengths(self):
        for size in (4, 5):
            with self.subTest(size=size):
                values = np.arange(size, dtype=np.float32)
                np.testing.assert_array_equal(fftshift1d(values), np.fft.fftshift(values))

    def test_subpixel_reshaping_preserves_legacy_channel_order(self):
        images = np.arange(120, dtype=np.float32).reshape(1, 4, 6, 5)
        expected = np.concatenate(
            [images[:, row::2, col::2] for row in range(2) for col in range(2)],
            axis=-1,
        )
        for data_format in ("channels_first", "channels_last"):
            with self.subTest(data_format=data_format):
                if data_format == "channels_first":
                    source = images.transpose(0, 3, 1, 2)
                    downscaled = expected.transpose(0, 3, 1, 2)
                else:
                    source, downscaled = images, expected
                np.testing.assert_array_equal(
                    space_to_depth(source, 2, data_format), downscaled
                )
                np.testing.assert_array_equal(
                    depth_to_space(downscaled, 2, data_format), source
                )

    def test_resize_images_aligns_corners_and_preserves_nearest_dtype(self):
        image = torch.tensor([[[[0, 30], [60, 90]]]], dtype=torch.uint8)
        nearest = resize_images(image, 2, 2, "nearest", "channels_first")
        self.assertEqual(nearest.dtype, torch.uint8)
        np.testing.assert_array_equal(
            nearest[0, 0],
            [[0, 0, 30, 30], [0, 0, 30, 30], [60, 60, 90, 90], [60, 60, 90, 90]],
        )
        for method in ("bilinear", "bicubic"):
            resized = resize_images(
                image.permute(0, 2, 3, 1), 2, 2, method, "channels_last"
            )
            np.testing.assert_allclose(resized[0, ::3, ::3, 0], image[0, 0])
        bilinear = resize_images(image, 2, 2, "bilinear", "channels_first")
        np.testing.assert_allclose(
            bilinear[0, 0],
            [[0, 10, 20, 30], [20, 30, 40, 50], [40, 50, 60, 70], [60, 70, 80, 90]],
            atol=1e-5,
        )

    def test_skeleton_parent_graph_resolves_valid_roots_and_rejects_cycles(self):
        graph = np.array([2, 0, -1, 2, 1, -1], dtype=np.float64)
        np.testing.assert_array_equal(graph_to_edges(graph), [2, 2, 2, 2, 2, 5])
        self.assertEqual(graph_to_edges(graph).dtype, graph.dtype)
        self.assertEqual(graph_to_edges([-1, 0]), [0, 0])
        for invalid in ([0], [1, 0, -1], [2, -1, 0]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "cycle"):
                    graph_to_edges(invalid)
        for invalid in ([2, -1], [-2], [0.5, -1], [np.nan]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "valid keypoint indices"):
                    graph_to_edges(invalid)

    def test_cyclic_annotation_skeleton_fails_before_a_training_batch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cyclic.h5"
            images = np.zeros((2, 32, 32, 1), dtype=np.uint8)
            points = np.tile([[[6, 8], [12, 14]]], (2, 1, 1))
            skeleton = pd.DataFrame(
                {
                    "name": ["left", "right"],
                    "parent": ["left", ""],
                    "swap": ["right", "left"],
                }
            )
            initialize_dataset(str(path), images, skeleton, points)
            with h5py.File(path, "r+") as file:
                file["annotated"][:] = True
            with self.assertRaisesRegex(ValueError, "cycle"):
                training = TrainingGenerator(DataGenerator(str(path)), use_graph=True)
                training(n_outputs=1, batch_size=1)[0]

    def test_train_save_reload_with_annotated_hdf5_and_keypoint_flip(self):
        keras.utils.set_random_seed(17)
        with TemporaryDirectory() as directory:
            annotation_path = Path(directory) / "annotations.h5"
            model_path = Path(directory) / "modern.keras"
            images = np.random.default_rng(17).integers(
                0, 256, size=(4, 64, 64, 1), dtype=np.uint8
            )
            keypoint_row = np.array([[8.0, 12.0], [31.0, 24.0]])
            keypoints = np.repeat(keypoint_row[None], len(images), axis=0)
            skeleton = pd.DataFrame(
                {
                    "name": ["left", "right"],
                    "parent": ["", "left"],
                    "swap": ["right", "left"],
                }
            )
            initialize_dataset(str(annotation_path), images, skeleton, keypoints)
            with h5py.File(annotation_path, "r+") as annotation_file:
                annotation_file["annotated"][:] = True

            annotations = DataGenerator(str(annotation_path))
            np.testing.assert_array_equal(annotations.get_images([0, 1]), images[:2])
            np.testing.assert_array_equal(
                annotations.get_keypoints([0, 1]), keypoints[:2]
            )
            np.testing.assert_array_equal(annotations.swap_index, [1, 0])

            training = TrainingGenerator(
                annotations,
                downsample_factor=2,
                use_graph=False,
                shuffle=False,
                augmenter=FlipAxis(annotations, axis=1, p=1.0, seed=17),
            )
            flipped_images, flipped_keypoints = training.augment(
                images[:1].copy(), keypoints[:1].copy()
            )
            np.testing.assert_array_equal(flipped_images, images[:1, :, ::-1])
            np.testing.assert_array_equal(
                flipped_keypoints, [[[32.0, 24.0], [55.0, 12.0]]]
            )

            model = StackedDenseNet(
                training, n_stacks=1, n_transitions=3, growth_rate=8, pretrained=False
            )
            optimizer = keras.optimizers.Adam()
            model.compile(optimizer=optimizer, loss="mse")
            self.assertEqual(int(optimizer.iterations.numpy()), 0)
            history = model.fit(batch_size=2, epochs=1, steps_per_epoch=1, verbose=0)
            self.assertEqual(int(optimizer.iterations.numpy()), 1)
            self.assertTrue(np.isfinite(history.history["loss"][0]))
            predictions = model.predict(images[:2], verbose=0)
            self.assertEqual(predictions.shape, (2, 2, 3))
            self.assertTrue(np.isfinite(predictions).all())

            model.save(model_path)
            reloaded = load_model(model_path, compile=False)
            np.testing.assert_allclose(
                reloaded.predict(images[:2], verbose=0),
                predictions,
                rtol=1e-6,
                atol=1e-6,
            )

            video_path = str(Path(directory) / "training-samples.avi")
            write_lossless_video(video_path, np.repeat(images[:2], 3, axis=-1))
            reader = VideoReader(video_path, batch_size=2, gray=True)
            try:
                for _ in range(2):
                    np.testing.assert_allclose(
                        reloaded.predict(reader, verbose=0),
                        predictions,
                        rtol=1e-6,
                        atol=1e-6,
                    )
            finally:
                reader.close()

    def test_ordinary_video_reader_returns_ordered_grayscale_frames(self):
        with TemporaryDirectory() as directory:
            video_path = str(Path(directory) / "frames.avi")
            frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
            frames[..., 0] = 7
            frames[..., 1] = np.array([10, 40, 80, 120, 160])[:, None, None]
            frames[..., 2] = 220
            write_lossless_video(video_path, frames)

            reader = VideoReader(video_path, batch_size=2, gray=True)
            try:
                self.assertEqual(len(reader), 3)
                np.testing.assert_array_equal(reader[0], frames[:2, ..., 1:2])
                np.testing.assert_array_equal(reader[2], frames[4:, ..., 1:2])
                np.testing.assert_array_equal(reader[1], frames[2:4, ..., 1:2])
            finally:
                reader.close()
            # Exercise the native capture lifecycle before the next model operation.
            del reader
            gc.collect()

    def test_fine_tune_imported_checkpoint_with_fresh_optimizer(self):
        reference_directory = os.environ.get("DPK_LEGACY_REFERENCE_DIR")
        if reference_directory is None:
            self.skipTest(
                "set DPK_LEGACY_REFERENCE_DIR to run the public-checkpoint test"
            )
        reference = Path(reference_directory)
        annotations = DataGenerator(str(reference / "annotation_data_release.h5"))
        model = load_model(
            reference / "best_model_densenet.h5", generator=annotations, compile=False
        )
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="mse")
        self.assertEqual(int(optimizer.iterations.numpy()), 0)
        history = model.fit(
            batch_size=1, epochs=1, steps_per_epoch=1, validation_steps=1, verbose=0
        )
        self.assertEqual(int(optimizer.iterations.numpy()), 1)
        self.assertTrue(np.isfinite(history.history["loss"][0]))
