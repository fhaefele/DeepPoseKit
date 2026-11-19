"""Synthetic legacy fixtures, run by the pinned capture_legacy_reference entry point."""

import hashlib
import json
import os
import platform
import random
import sys

# Small configurations retain each upstream model's layers and decoder.
BACKBONES = {
    "LEAP": {
        "training": {"downsample_factor": 0, "use_graph": False, "sigma": 3},
        "model": {
            "filters": 8,
            "batchnorm": True,
            "pooling": "average",
            "upsampling": True,
            "interpolation": "bilinear",
            "subpixel": False,
        },
    },
    "DeepLabCut": {
        "training": {"downsample_factor": 2, "use_graph": False, "sigma": 5},
        "model": {
            "backbone": "mobilenetv2",
            "alpha": 0.35,
            "weights": None,
            "subpixel": True,
        },
    },
    "StackedHourglass": {
        "training": {"downsample_factor": 2, "use_graph": False, "sigma": 5},
        "model": {
            "n_stacks": 2,
            "n_transitions": 3,
            "filters": 16,
            "bottleneck_factor": 2,
            "subpixel": True,
        },
    },
}

# Set before comparing the legacy and modern runtimes, as for the fly reference.
PARITY_TOLERANCE = {
    "absolute_heatmap": 0.001,
    "keypoint_euclidean_pixels": 0.06,
    "absolute_confidence": 0.00001,
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_annotations(path):
    """Write identical lossless training inputs in the old and modern runtimes."""
    import h5py
    import numpy as np
    import pandas as pd

    from deepposekit.io import initialize_dataset

    images = np.random.RandomState(23).randint(
        0, 256, size=(4, 64, 64, 1), dtype=np.uint8
    )
    keypoints = np.tile([[[12.0, 20.0], [44.0, 36.0]]], (4, 1, 1))
    skeleton = pd.DataFrame(
        {
            "name": ["left", "right"],
            "parent": ["", "left"],
            "swap": ["right", "left"],
        }
    )
    initialize_dataset(str(path), images, skeleton, keypoints)
    with h5py.File(path, "r+") as file:
        file["annotated"][:] = True
    return images


def capture_backbones(source, output, names, source_revision):
    """Capture through upstream APIs without changing the pinned source checkout."""
    unknown = set(names) - BACKBONES.keys()
    if unknown:
        raise ValueError(f"Unsupported backbone fixtures: {sorted(unknown)}")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(output / ".matplotlib"),
        OMP_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))

    from importlib.metadata import version

    import h5py
    import numpy as np
    import tensorflow as tf

    from deepposekit import models
    from deepposekit.io import DataGenerator, TrainingGenerator

    for name in names:
        tf.keras.backend.clear_session()
        random.seed(23)
        np.random.seed(23)
        tf.random.set_seed(23)
        annotations = output / f"{name}-annotations.h5"
        images = create_annotations(annotations)
        settings = BACKBONES[name]
        training = TrainingGenerator(
            DataGenerator(str(annotations)), shuffle=False, **settings["training"]
        )
        model = getattr(models, name)(training, **settings["model"])
        model.compile(optimizer="adam", loss="mse")
        batch = training(n_outputs=model.n_outputs, batch_size=2)[0]
        loss = model.train_model.train_on_batch(*batch)
        if not np.isfinite(loss).all():
            raise ValueError(f"Non-finite legacy training loss for {name}: {loss}")
        heatmaps = model.train_model.predict(images, batch_size=1)
        if isinstance(heatmaps, list):
            heatmaps = heatmaps[-1]
        predictions = model.predict(images, batch_size=1)
        repeated = model.predict(images, batch_size=1)
        checkpoint = output / f"{name}.h5"
        reference = output / f"{name}-reference.npz"
        model.save(str(checkpoint))
        np.savez_compressed(
            reference, frames=images, heatmaps=heatmaps, predictions=predictions
        )
        with h5py.File(checkpoint, "r") as file:
            writer = file.attrs["keras_version"]
            if isinstance(writer, bytes):
                writer = writer.decode("utf-8")
        manifest = {
            "source_revision": source_revision,
            "model": name,
            "configuration": settings,
            "seed": 23,
            "training_updates": 1,
            "training_loss": np.asarray(loss).tolist(),
            "checkpoint_keras_version": writer,
            "runtime": {
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "tensorflow-cpu": version("tensorflow-cpu"),
                "numpy": version("numpy"),
            },
            "output_sha256": {
                path.name: _sha256(path)
                for path in (checkpoint, reference, annotations)
            },
            "frame_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
            "preprocessing": "Synthetic uint8 grayscale frames passed unchanged to both runtimes",
            "parity_tolerance": PARITY_TOLERANCE,
            "max_repeat_prediction_error": float(
                np.max(np.abs(predictions - repeated))
            ),
        }
        (output / f"{name}-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "model": name,
                    "training_loss": manifest["training_loss"],
                    "max_repeat_prediction_error": manifest[
                        "max_repeat_prediction_error"
                    ],
                }
            )
        )
