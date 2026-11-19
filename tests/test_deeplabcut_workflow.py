import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import keras
import numpy as np

from deepposekit.io import TrainingGenerator
from deepposekit.io.BaseGenerator import BaseGenerator
from deepposekit.models import DeepLabCut, load_model


class MemoryAnnotations(BaseGenerator):
    def __init__(self, channels):
        self.images = np.random.default_rng(10).integers(
            0, 256, size=(4, 96, 96, channels), dtype=np.uint8
        )
        self.keypoints = np.tile([[18.0, 22.0], [60.0, 47.0]], (4, 1, 1))
        super().__init__()

    def __len__(self):
        return len(self.images)

    def compute_image_shape(self):
        return self.images.shape[1:]

    def compute_keypoints_shape(self):
        return self.keypoints.shape[1:]

    def get_images(self, indexes):
        return self.images[indexes]

    def get_keypoints(self, indexes):
        return self.keypoints[indexes]


class DeepLabCutWorkflowTests(unittest.TestCase):
    def test_original_encoders_and_stride_heads_on_cpu(self):
        backbones = (
            "resnet50",
            "resnet101",
            "resnet152",
            "densenet121",
            "densenet169",
            "densenet201",
            "mobilenetv2",
            "xception",
        )
        for downsample_factor in (2, 3, 4):
            annotations = MemoryAnnotations(channels=1 if downsample_factor == 3 else 3)
            generator = TrainingGenerator(
                annotations,
                downsample_factor=downsample_factor,
                use_graph=False,
                validation_split=0.25,
                shuffle=False,
            )
            size = 96 // (2**downsample_factor)
            for backbone in backbones:
                with self.subTest(
                    backbone=backbone, output_stride=2**downsample_factor
                ):
                    model = DeepLabCut(
                        generator,
                        backbone=backbone,
                        alpha=0.35,
                        weights=None,
                    )
                    frames = annotations.images[:1]
                    heatmaps = model.train_model(frames, training=False).numpy()
                    predictions = model.predict_model(frames, training=False).numpy()
                    self.assertEqual(heatmaps.shape, (1, size, size, 2))
                    self.assertEqual(predictions.shape, (1, 2, 3))
                    self.assertTrue(np.isfinite(heatmaps).all())
                    self.assertTrue(np.isfinite(predictions).all())
                    del model
                keras.utils.clear_session()

    def test_train_and_reload_stride_sixteen_mobilenet(self):
        keras.utils.set_random_seed(10)
        annotations = MemoryAnnotations(channels=1)
        generator = TrainingGenerator(
            annotations,
            downsample_factor=4,
            use_graph=False,
            validation_split=0.25,
            shuffle=False,
        )
        backbone = "".join(("mobile", "netv2"))
        model = DeepLabCut(generator, backbone=backbone, alpha=0.35, weights=None)
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="mse")
        history = model.fit(batch_size=1, epochs=1, steps_per_epoch=1, verbose=0)
        self.assertEqual(int(optimizer.iterations.numpy()), 1)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

        frames = annotations.images[:1]
        predictions = model.predict_on_batch(frames)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "deeplabcut.keras"
            model.save(checkpoint)
            reloaded = load_model(checkpoint, compile=False)
            self.assertIsInstance(reloaded, DeepLabCut)
            self.assertEqual(reloaded.backbone, "mobilenetv2")
            self.assertEqual(reloaded.alpha, 0.35)
            np.testing.assert_allclose(
                reloaded.predict_on_batch(frames), predictions, rtol=1e-6, atol=1e-6
            )
