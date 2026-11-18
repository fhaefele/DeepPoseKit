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

import h5py
import keras
import numpy as np

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
        with np.load(cls.reference, allow_pickle=False) as data:
            cls.frames = data["frames"]
            cls.expected = {name: data[name] for name in ("heatmaps", "predictions")}

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
            self.assertLessEqual(score, self.manifest["parity_tolerance"][name], name)

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


if __name__ == "__main__":
    _checkpoint_prediction(*sys.argv[1:])
