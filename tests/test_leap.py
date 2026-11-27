import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from backbone_test_utils import BackboneWorkflowMixin
from fixtures.backbone_reference import create_annotations

from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import LEAP, load_model

import keras


class LEAPWorkflowTest(BackboneWorkflowMixin, unittest.TestCase):
    backbone_name = "LEAP"

    def test_transposed_convolution_and_callable_activation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations.h5"
            frames = create_annotations(annotations)
            training = TrainingGenerator(
                DataGenerator(str(annotations)), downsample_factor=0, use_graph=False
            )
            model = LEAP(
                training, filters=8, activation=keras.activations.selu, batchnorm=True
            )
            layers = model.train_model.layers
            self.assertEqual(
                sum(
                    isinstance(layer, keras.layers.Conv2DTranspose) for layer in layers
                ),
                2,
            )
            self.assertFalse(
                any(
                    isinstance(layer, keras.layers.BatchNormalization)
                    for layer in layers
                )
            )
            outputs = model.predict(frames, batch_size=1, verbose=0)
            self.assertEqual(outputs.shape, (4, 2, 3))
            self.assertTrue(np.isfinite(outputs).all())
            checkpoint = root / "transposed.keras"
            model.save(checkpoint)
            np.testing.assert_allclose(
                load_model(checkpoint, compile=False).predict(
                    frames, batch_size=1, verbose=0
                ),
                outputs,
            )
