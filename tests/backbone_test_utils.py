"""Common public-API workflow and legacy parity checks for migrated backbones."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fixtures.backbone_reference import BACKBONES, create_annotations

from deepposekit import models
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import load_model

import keras

SOURCE_REVISION = "cecdb0c8c364ea049a3b705275ae71a2f366d4da"


class BackboneWorkflowMixin:
    """Set backbone_name on a unittest.TestCase subclass to verify a port."""

    def test_modern_training_save_reload(self):
        keras.utils.set_random_seed(23)
        settings = BACKBONES[self.backbone_name]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            annotations = directory / "annotations.h5"
            images = create_annotations(annotations)
            training = TrainingGenerator(
                DataGenerator(str(annotations)), shuffle=False, **settings["training"]
            )
            model = getattr(models, self.backbone_name)(training, **settings["model"])
            optimizer = keras.optimizers.Adam()
            model.compile(optimizer=optimizer, loss="mse")
            history = model.fit(batch_size=2, epochs=1, steps_per_epoch=1, verbose=0)
            self.assertEqual(int(optimizer.iterations.numpy()), 1)
            self.assertTrue(np.isfinite(history.history["loss"]).all())
            predicted = model.predict(images, batch_size=1, verbose=0)
            self.assertEqual(predicted.shape, (4, 2, 3))
            self.assertTrue(np.isfinite(predicted).all())
            checkpoint = directory / "modern.keras"
            model.save(checkpoint)
            restored = load_model(checkpoint, compile=False)
            np.testing.assert_allclose(
                restored.predict(images, batch_size=1, verbose=0),
                predicted,
                rtol=1e-6,
                atol=1e-6,
            )

    def _assert_reference_parity(self, model, reference, manifest, label):
        images = reference["frames"]
        heatmaps = model.train_model.predict(images, batch_size=1, verbose=0)
        if isinstance(heatmaps, list):
            heatmaps = heatmaps[-1]
        predictions = model.predict(images, batch_size=1, verbose=0)
        self.assertEqual(heatmaps.shape, reference["heatmaps"].shape)
        self.assertEqual(predictions.shape, reference["predictions"].shape)
        delta = predictions - reference["predictions"]
        errors = {
            "absolute_heatmap": float(np.max(np.abs(heatmaps - reference["heatmaps"]))),
            "keypoint_euclidean_pixels": float(
                np.max(np.linalg.norm(delta[..., :2], axis=-1))
            ),
            "absolute_confidence": float(np.max(np.abs(delta[..., 2]))),
        }
        print(
            f"{self.backbone_name} {label} {json.dumps(errors, sort_keys=True)}",
            flush=True,
        )
        for metric, error in errors.items():
            self.assertTrue(np.isfinite(error), metric)
            self.assertLessEqual(error, manifest["parity_tolerance"][metric], metric)

    def test_legacy_import_reload_and_fine_tune(self):
        root = os.environ.get("DPK_LEGACY_REFERENCE_DIR")
        if root is None:
            self.skipTest(
                "set DPK_LEGACY_REFERENCE_DIR to the pinned backbone fixtures"
            )
        root = Path(root)
        manifest = json.loads(
            (root / f"{self.backbone_name}-manifest.json").read_text()
        )
        self.assertEqual(manifest["source_revision"], SOURCE_REVISION)
        self.assertEqual(manifest["checkpoint_keras_version"], "2.3.0-tf")
        self.assertEqual(manifest["training_updates"], 1)
        self.assertEqual(manifest["max_repeat_prediction_error"], 0)
        for name, expected in manifest["output_sha256"].items():
            with (root / name).open("rb") as file:
                self.assertEqual(
                    hashlib.file_digest(file, "sha256").hexdigest(), expected, name
                )
        with (
            np.load(
                root / f"{self.backbone_name}-reference.npz", allow_pickle=False
            ) as reference,
            TemporaryDirectory() as temporary,
        ):
            directory = Path(temporary)
            legacy = directory / "relocated.hdf5"
            shutil.copyfile(root / f"{self.backbone_name}.h5", legacy)
            model = load_model(legacy, compile=False)
            self._assert_reference_parity(model, reference, manifest, "legacy")
            modern = directory / "imported.keras"
            model.save(modern)
            restored = load_model(modern, compile=False)
            self._assert_reference_parity(
                restored, reference, manifest, "modern-reload"
            )

            annotations = DataGenerator(
                str(root / f"{self.backbone_name}-annotations.h5")
            )
            trainable = load_model(modern, generator=annotations, compile=False)
            optimizer = keras.optimizers.Adam()
            trainable.compile(optimizer=optimizer, loss="mse")
            history = trainable.fit(
                batch_size=2, epochs=1, steps_per_epoch=1, verbose=0
            )
            self.assertEqual(int(optimizer.iterations.numpy()), 1)
            self.assertTrue(np.isfinite(history.history["loss"]).all())
