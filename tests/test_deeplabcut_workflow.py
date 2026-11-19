import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import keras
import numpy as np

from deepposekit.io import TrainingGenerator
from deepposekit.io.BaseGenerator import BaseGenerator
from deepposekit.models import DeepLabCut, load_model
from deepposekit.models._checkpoint import assign_legacy_weights


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
            for suffix in (".keras", ".h5"):
                with self.subTest(suffix=suffix):
                    checkpoint = Path(directory) / f"deeplabcut{suffix}"
                    model.save(checkpoint)
                    reloaded = load_model(checkpoint, compile=False)
                    self.assertIsInstance(reloaded, DeepLabCut)
                    self.assertEqual(reloaded.backbone, "mobilenetv2")
                    self.assertEqual(reloaded.alpha, 0.35)
                    np.testing.assert_allclose(
                        reloaded.predict_on_batch(frames),
                        predictions,
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_legacy_depthwise_kernel_role_is_specific_and_unambiguous(self):
        inputs = keras.Input((4, 4, 1))
        depthwise = keras.layers.DepthwiseConv2D(1, use_bias=False, name="depthwise")
        pointwise = keras.layers.Conv2D(1, 1, use_bias=False, name="pointwise")
        encoder = keras.Model(inputs, pointwise(depthwise(inputs)), name="encoder")
        model_inputs = keras.Input((4, 4, 1))
        model = keras.Model(model_inputs, encoder(model_inputs))

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.h5"
            with h5py.File(checkpoint, "w") as file:
                file.attrs["keras_version"] = "2.3.0-tf"
                file.attrs["layer_names"] = ["encoder"]
                group = file.create_group("encoder")
                depthwise_identity = "depthwise_1/depthwise_kernel:0"
                pointwise_identity = "pointwise_1/kernel:0"
                group.attrs["weight_names"] = [depthwise_identity, pointwise_identity]
                for layer, identity in (
                    (depthwise, depthwise_identity),
                    (pointwise, pointwise_identity),
                ):
                    group.create_dataset(
                        identity, data=np.ones(layer.kernel.shape, dtype=np.float32)
                    )

                assign_legacy_weights(model, file)
                np.testing.assert_array_equal(depthwise.kernel.numpy(), 1)
                np.testing.assert_array_equal(pointwise.kernel.numpy(), 1)

                incorrect_role = "pointwise_1/depthwise_kernel:0"
                group.move(pointwise_identity, incorrect_role)
                group.attrs["weight_names"] = [depthwise_identity, incorrect_role]
                with self.assertRaisesRegex(ValueError, "pointwise/kernel"):
                    assign_legacy_weights(model, file)
                np.testing.assert_array_equal(pointwise.kernel.numpy(), 1)

                duplicate_role = "depthwise/kernel:0"
                group.copy(depthwise_identity, duplicate_role)
                group.attrs["weight_names"] = [
                    depthwise_identity,
                    incorrect_role,
                    duplicate_role,
                ]
                with self.assertRaisesRegex(
                    ValueError, "Duplicate legacy tensor identity"
                ):
                    assign_legacy_weights(model, file)
