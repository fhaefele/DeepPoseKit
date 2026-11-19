"""End-to-end parity with the pinned upstream StackedDenseNet reference."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import keras
import numpy as np
from fixtures.capture_legacy_reference import (
    ASSET_HASHES,
    DATA_REVISION,
    FRAME_INDICES,
    PARITY_TOLERANCE,
    SOURCE_REVISION,
)

from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import StackedDenseNet, load_legacy_weights, load_model

REFERENCE_DIR = os.environ.get("DPK_LEGACY_REFERENCE_DIR")


def _predict(model, frames):
    return {
        "heatmaps": model.train_model.predict(frames, batch_size=1, verbose=0)[-1],
        "predictions": model.predict(frames, batch_size=1, verbose=0),
    }


def _checkpoint_prediction(checkpoint, reference, output, save_to=None):
    with np.load(reference, allow_pickle=False) as data:
        frames = data["frames"]
    model = load_model(checkpoint, compile=False)
    np.savez(output, **_predict(model, frames))
    if save_to is not None:
        model.save(save_to)


def _sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _verify_reference_manifest(manifest):
    expected = {
        "source_revision": SOURCE_REVISION,
        "data_revision": DATA_REVISION,
        "asset_sha256": ASSET_HASHES,
        "frame_indices": list(FRAME_INDICES),
        "checkpoint_keras_version": "2.2.4-tf",
        "weights_keras_version": "2.3.0-tf",
        "parity_tolerance": PARITY_TOLERANCE,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Legacy reference manifest has unexpected {field}")

    repeat_errors = manifest.get("max_repeat_error", {})
    if not isinstance(repeat_errors, dict):
        raise TypeError("Legacy reference repeatability measurements must be a mapping")
    for metric, tolerance in PARITY_TOLERANCE.items():
        error = repeat_errors.get(metric)
        if type(error) not in (int, float) or not 0 <= error <= tolerance:
            raise ValueError(
                f"Legacy reference repeatability check failed for {metric}"
            )


def _load_reference_archive(path, manifest):
    if _sha256(path) != manifest["output_sha256"].get(path.name):
        raise ValueError(f"Reference archive SHA-256 mismatch for {path}")
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"]
        expected = {name: data[name] for name in ("heatmaps", "predictions")}
    if hashlib.sha256(frames.tobytes()).hexdigest() != manifest.get("frame_sha256"):
        raise ValueError(f"Reference frame SHA-256 mismatch for {path}")
    return frames, expected


class ReferenceProvenanceTest(unittest.TestCase):
    def test_rejects_mismatched_reference_archive_and_frames(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.npz"
            frames = np.arange(4, dtype=np.uint8).reshape(1, 2, 2, 1)
            outputs = {
                "heatmaps": np.zeros((1, 2, 2, 1), dtype=np.float32),
                "predictions": np.zeros((1, 1, 3), dtype=np.float32),
            }
            np.savez(path, frames=frames, **outputs)
            manifest = {
                "output_sha256": {path.name: _sha256(path)},
                "frame_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            }
            actual_frames, _ = _load_reference_archive(path, manifest)
            np.testing.assert_array_equal(actual_frames, frames)

            np.savez(path, frames=np.zeros_like(frames), **outputs)
            with self.assertRaisesRegex(ValueError, "archive SHA-256 mismatch"):
                _load_reference_archive(path, manifest)
            manifest["output_sha256"][path.name] = _sha256(path)
            with self.assertRaisesRegex(ValueError, "frame SHA-256 mismatch"):
                _load_reference_archive(path, manifest)


@unittest.skipUnless(REFERENCE_DIR, "requires the pinned legacy reference fixtures")
class LegacyStackedDenseNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(REFERENCE_DIR)
        cls.full = cls.directory / "best_model_densenet.h5"
        cls.weights = cls.directory / "legacy.weights-only.h5"
        cls.reference = cls.directory / "fly-reference.npz"
        cls.manifest = json.loads(
            (cls.directory / "manifest.json").read_text(encoding="utf-8")
        )
        _verify_reference_manifest(cls.manifest)
        cls.frames, cls.expected = _load_reference_archive(cls.reference, cls.manifest)
        for checkpoint, field in (
            (cls.full, "checkpoint_keras_version"),
            (cls.weights, "weights_keras_version"),
        ):
            with h5py.File(checkpoint, "r") as file:
                writer = file.attrs.get("keras_version")
            if isinstance(writer, bytes):
                writer = writer.decode("utf-8")
            if writer != cls.manifest[field]:
                raise ValueError(f"Legacy Keras writer mismatch for {checkpoint}")

    def _assert_parity(self, case, actual):
        outputs = self.expected
        for name in outputs:
            self.assertEqual(actual[name].shape, outputs[name].shape, name)
        error = actual["predictions"] - outputs["predictions"]
        scores = {
            "absolute_heatmap": float(
                np.max(np.abs(actual["heatmaps"] - outputs["heatmaps"]))
            ),
            "keypoint_euclidean_pixels": float(
                np.max(np.linalg.norm(error[..., :2], axis=-1))
            ),
            "absolute_confidence": float(np.max(np.abs(error[..., 2]))),
        }
        print(f"{case} {json.dumps(scores, sort_keys=True)}", flush=True)
        for name, score in scores.items():
            self.assertLessEqual(score, PARITY_TOLERANCE[name], name)

    def _fresh_process_prediction(self, checkpoint, directory, label, save_to=None):
        cache = directory / f"keras-{label}"
        output = directory / f"{label}.npz"
        self.assertFalse(cache.exists())
        arguments = [str(checkpoint), str(self.reference), str(output)]
        if save_to is not None:
            arguments.append(str(save_to))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            env={**os.environ, "KERAS_HOME": str(cache)},
            check=True,
        )
        with np.load(output, allow_pickle=False) as actual:
            self._assert_parity(label, actual)

    def test_full_checkpoint_is_portable(self):
        original_annotations = self.manifest["training_generator"]["config"]["datapath"]
        self.assertFalse(Path(original_annotations).exists())
        checksum = self.manifest["asset_sha256"][self.full.name]
        self.assertEqual(_sha256(self.full), checksum)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "relocated.hdf5"
            shutil.copyfile(self.full, checkpoint)
            modern = directory / "fly.keras"
            self._fresh_process_prediction(checkpoint, directory, "legacy-full", modern)
            self._fresh_process_prediction(modern, directory, "modern-full", None)
            self.assertEqual(_sha256(checkpoint), checksum)
        self.assertEqual(_sha256(self.full), checksum)

    def test_incomplete_checkpoint_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "weights-only"):
            load_model(self.weights, compile=False)
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "missing-metadata.h5"
            shutil.copyfile(self.full, checkpoint)
            with h5py.File(checkpoint, "r+") as file:
                del file.attrs["pose_model_config"]
            with self.assertRaisesRegex(ValueError, "pose_model_config"):
                load_model(checkpoint, compile=False)

            with h5py.File(checkpoint, "r+") as file:
                pose = self.manifest["pose_model"]
                config = {**pose["config"], "keypoints_shape": [31, 2]}
                file.attrs["pose_model_config"] = json.dumps({**pose, "config": config})
            with self.assertRaisesRegex(ValueError, "metadata disagree"):
                load_model(checkpoint, compile=False)

    def test_decoder_metadata_controls_coordinate_scaling(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "decoder.h5"
            shutil.copyfile(self.full, checkpoint)
            with h5py.File(checkpoint, "r+") as file:
                pose = json.loads(file.attrs["pose_model_config"])
                generator = json.loads(file.attrs["train_generator_config"])
                pose["config"]["subpixel"] = False
                file.attrs["pose_model_config"] = json.dumps(pose)

            model = load_model(checkpoint, compile=False)
            frames = self.frames[:1]
            heatmaps = model.train_model.predict_on_batch(frames)[-1]
            keypoints = heatmaps[..., : pose["config"]["keypoints_shape"][0]]
            stride = 2 ** pose["config"]["downsample_factor"]
            expected = np.stack(
                (
                    keypoints.max(axis=1).argmax(axis=1) * stride,
                    keypoints.max(axis=2).argmax(axis=1) * stride,
                    keypoints.max(axis=(1, 2)) / 255.0,
                ),
                axis=-1,
            )
            np.testing.assert_allclose(
                model.predict_on_batch(frames), expected, rtol=0, atol=1e-6
            )
            modern = Path(temporary) / "integer-decoder.keras"
            model.save(modern)
            restored = load_model(modern, compile=False)
            np.testing.assert_allclose(
                restored.predict_on_batch(frames), expected, rtol=0, atol=1e-6
            )

            for value in (None, "false", 0):
                with self.subTest(subpixel=value):
                    if value is None:
                        del pose["config"]["subpixel"]
                    else:
                        pose["config"]["subpixel"] = value
                    with h5py.File(checkpoint, "r+") as file:
                        file.attrs["pose_model_config"] = json.dumps(pose)
                    with self.assertRaisesRegex(ValueError, "subpixel"):
                        load_model(checkpoint, compile=False)

            pose["config"]["subpixel"] = True
            for metadata in (pose, generator):
                metadata["config"]["downsample_factor"] -= 1
            with h5py.File(checkpoint, "r+") as file:
                file.attrs["pose_model_config"] = json.dumps(pose)
                file.attrs["train_generator_config"] = json.dumps(generator)
            with self.assertRaisesRegex(ValueError, "downsample_factor"):
                load_model(checkpoint, compile=False)

    def _configured_model(self):
        # Exercise normal construction after the Keras global name counters advance.
        keras.layers.Conv2D(1, 1)
        keras.layers.BatchNormalization()
        annotations = DataGenerator(str(self.directory / "annotation_data_release.h5"))
        generator_settings = self.manifest["training_generator"]["config"]
        generator = TrainingGenerator(
            annotations,
            downsample_factor=generator_settings["downsample_factor"],
            use_graph=generator_settings["use_graph"],
            sigma=generator_settings["sigma"],
            validation_split=generator_settings["validation_split"],
        )
        config = self.manifest["pose_model"]["config"]
        keys = (
            "n_stacks",
            "n_transitions",
            "growth_rate",
            "bottleneck_factor",
            "compression_factor",
            "pretrained",
            "subpixel",
        )
        return StackedDenseNet(
            generator, pretrained_weights=None, **{key: config[key] for key in keys}
        )

    def test_weights_only_predictions_and_atomic_validation(self):
        checksum = self.manifest["output_sha256"][self.weights.name]
        self.assertEqual(_sha256(self.weights), checksum)
        model = self._configured_model()
        load_legacy_weights(model, self.weights)
        self._assert_parity("legacy-weights-only", _predict(model, self.frames))
        expected = model.predict_on_batch(self.frames[:1])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            modern = directory / "weights-import.keras"
            model.save(modern)
            restored = load_model(modern, compile=False)
            self._assert_parity("modern-weights-only", _predict(restored, self.frames))

            # Use a zero-initialized target to detect any partial assignments.
            for variable in model.train_model.weights:
                variable.assign(np.zeros(variable.shape, dtype=variable.dtype))
            original = [variable.numpy() for variable in model.train_model.weights]
            original_prediction = model.predict_on_batch(self.frames[:1])

            for problem in ("shape", "identity"):
                with self.subTest(problem=problem):
                    broken = directory / f"{problem}.h5"
                    shutil.copyfile(self.weights, broken)
                    with h5py.File(broken, "r+") as file:
                        group = file["output_2"]
                        bias = "output_2/bias:0"
                        if problem == "shape":
                            values = group[bias][()]
                            del group[bias]
                            group.create_dataset(bias, data=values[:-1])
                        else:
                            unknown = "output_2/unknown:0"
                            group.move(bias, unknown)
                            names = [
                                unknown.encode() if name == bias.encode() else name
                                for name in group.attrs["weight_names"]
                            ]
                            group.attrs["weight_names"] = np.asarray(names)
                    with self.assertRaisesRegex(
                        ValueError, "identit" if problem == "identity" else problem
                    ):
                        load_legacy_weights(model, broken)
                    for variable, value in zip(model.train_model.weights, original):
                        np.testing.assert_array_equal(variable.numpy(), value)
                    np.testing.assert_array_equal(
                        model.predict_on_batch(self.frames[:1]), original_prediction
                    )

            load_legacy_weights(model, self.weights)
            np.testing.assert_array_equal(
                model.predict_on_batch(self.frames[:1]), expected
            )
        self.assertEqual(_sha256(self.weights), checksum)

    def test_backend_assignment_failure_restores_weights_and_predictions(self):
        model = self._configured_model()
        variables = model.train_model.weights
        originals = [variable.numpy() for variable in variables]
        frames = self.frames[:1]
        prediction = model.predict_on_batch(frames)
        last_variable = variables[-1]
        assign = last_variable.assign
        attempts = 0

        def fail_after_first_write(value):
            nonlocal attempts
            attempts += 1
            result = assign(value)
            if attempts == 1:
                raise RuntimeError("backend assignment interrupted")
            return result

        with patch.object(last_variable, "assign", side_effect=fail_after_first_write):
            with self.assertRaisesRegex(RuntimeError, "backend assignment interrupted"):
                load_legacy_weights(model, self.weights)

        self.assertEqual(attempts, 2)
        for variable, original in zip(variables, originals):
            np.testing.assert_array_equal(variable.numpy(), original)
        np.testing.assert_array_equal(model.predict_on_batch(frames), prediction)
        self.assertEqual(
            _sha256(self.weights), self.manifest["output_sha256"][self.weights.name]
        )


if __name__ == "__main__":
    _checkpoint_prediction(*sys.argv[1:])
