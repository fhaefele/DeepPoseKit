# /// script
# requires-python = "==3.8.20"
# dependencies = [
#     "h5py==2.10.0",
#     "imgaug==0.4.0",
#     "matplotlib==3.3.4",
#     "numpy==1.18.5",
#     "opencv-python==5.1.0.95",
#     "pandas==1.2.5",
#     "protobuf==3.20.3",
#     "pyyaml==6.0.3",
#     "scikit-image==0.17.2",
#     "scikit-learn==0.24.2",
#     "scipy==1.4.1",
#     "tensorflow-cpu==2.2.3",
# ]
# ///
"""Capture the CPU reference for the original DeepPoseKit fly checkpoint.

Run with uv run --locked --script tests/fixtures/capture_legacy_reference.py
--source /path/to/pinned/legacy/checkout --output /path/to/reference-directory.
The output directory can contain pre-staged public assets for offline capture.
The legacy checkout must have the pinned DeepPoseKit source without local edits.
Add --scoped-weights to capture the repeated-model fixture instead.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

SOURCE_REVISION = "cecdb0c8c364ea049a3b705275ae71a2f366d4da"
DATA_REVISION = "0aa5e3f5e8f9df63c48ba2bf491354472daa3e7e"
ASSET_URL = (
    "https://raw.githubusercontent.com/jgraving/DeepPoseKit-Data/"
    + DATA_REVISION
    + "/datasets/fly/"
)
ASSET_HASHES = {
    "best_model_densenet.h5": "1fdf7126550e3f0d3c0dea26c4987cd44bcd46e561f6e76622525fb240b658c8",
    "video.avi": "c2aea283d7b4fc3be24cf5ee1bb06ccf2720084032c7176edbce4a03c6eac040",
}
FRAME_INDICES = (0, 37, 419, 3203, 8049, 15121, 24218, 32832)
# Allow small float32 kernel differences across TensorFlow CPU versions.
# At stride 8 and 100x subpixel upsampling, one cell along each axis is
# about 0.113 px. Confidence is the maximum heatmap value scaled by 255.
PARITY_TOLERANCE = {
    "absolute_heatmap": 0.01,
    "keypoint_euclidean_pixels": 0.12,
    "absolute_confidence": 0.00005,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(file):
    import h5py

    digest = hashlib.sha256()

    def add_tensor(name, node):
        if isinstance(node, h5py.Dataset):
            header = json.dumps([name, node.shape, node.dtype.str]) + "\n"
            digest.update(header.encode("utf-8"))
            digest.update(node[()].tobytes())

    file.visititems(add_tensor)
    return digest.hexdigest()


def stage_asset(output, name, expected_hash):
    path = output / name
    if path.exists():
        if sha256(path) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {path}")
        return path

    with tempfile.TemporaryDirectory(dir=output) as staging:
        download = Path(staging) / name
        url = ASSET_URL + name
        with urlopen(url, timeout=60) as response, download.open("wb") as file:
            shutil.copyfileobj(response, file)
        if sha256(download) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for downloaded {name}")
        download.replace(path)
    return path


def check_source(source):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected legacy source {SOURCE_REVISION}, got {revision}")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "deepposekit"],
        check=True,
    )


def _configure_legacy_runtime(source, output):
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


def capture_scoped_weights(source, output):
    """Export the second StackedDenseNet built in a single legacy process."""
    from functools import partial
    from unittest.mock import patch

    output.mkdir(parents=True, exist_ok=True)
    _configure_legacy_runtime(source, output)

    import h5py
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from deepposekit.io import DataGenerator, TrainingGenerator, initialize_dataset
    from deepposekit.models import StackedDenseNet
    from deepposekit.models.layers.imagenet_densenet import DenseNet121

    frames = np.random.RandomState(19).randint(0, 256, (4, 64, 96, 1), dtype=np.uint8)
    skeleton = pd.DataFrame(
        {"name": ["left", "right"], "parent": ["", "left"], "swap": ["right", "left"]}
    )
    keypoints = np.tile([[[10.0, 12.0], [27.0, 32.0]]], (len(frames), 1, 1))
    annotation_path = str(output / "scoped-annotations.h5")
    initialize_dataset(annotation_path, frames, skeleton, keypoints, overwrite=True)
    with h5py.File(annotation_path, "r+") as annotations:
        annotations["annotated"][:] = True
    training = TrainingGenerator(
        DataGenerator(annotation_path), downsample_factor=3, use_graph=False
    )
    options = {
        "n_stacks": 1,
        "n_transitions": 4,
        "growth_rate": 4,
        "pretrained": True,
        "subpixel": False,
    }
    tf.random.set_seed(19)

    # Keep the ImageNet frontend architecture and build without downloading weights.
    frontend = partial(DenseNet121, weights=None)
    with patch("deepposekit.models.layers.densenet.DenseNet121", frontend):
        StackedDenseNet(training, **options)
        model = StackedDenseNet(training, **options)

    model.save(str(output / "scoped-full.h5"))
    model.train_model.save_weights(str(output / "scoped-weights.h5"), save_format="h5")
    np.savez_compressed(
        output / "scoped-reference.npz",
        frames=frames,
        heatmaps=model.train_model.predict(frames, batch_size=1, verbose=0)[-1],
        predictions=model.predict(frames, batch_size=1, verbose=0),
    )


def capture(source, output):
    from importlib.metadata import version

    _configure_legacy_runtime(source, output)

    import h5py
    import numpy as np

    from deepposekit.io import VideoReader
    from deepposekit.models import load_model

    checkpoint = stage_asset(
        output, "best_model_densenet.h5", ASSET_HASHES["best_model_densenet.h5"]
    )
    video = stage_asset(output, "video.avi", ASSET_HASHES["video.avi"])

    reader = VideoReader(str(video), gray=True)
    frames = []
    frame_indices = set(FRAME_INDICES)
    try:
        for index in range(FRAME_INDICES[-1] + 1):
            frame = reader.read()
            if frame is None:
                raise ValueError(f"Unexpected end of video at frame {index}")
            if index in frame_indices:
                frames.append(frame)
    finally:
        reader.close()
    frames = np.stack(frames)

    with h5py.File(checkpoint, "r") as file:
        producer = file.attrs["keras_version"]
        if isinstance(producer, bytes):
            producer = producer.decode("utf-8")
        model_config = json.loads(file.attrs["pose_model_config"])
        generator_config = json.loads(file.attrs["train_generator_config"])
    model = load_model(str(checkpoint), compile=False)
    heatmaps = model.train_model.predict(frames, batch_size=1)[-1]
    predictions = model.predict(frames, batch_size=1)
    repeat_heatmaps = model.train_model.predict(frames, batch_size=1)[-1]
    repeat_predictions = model.predict(frames, batch_size=1)
    coordinate_delta = predictions[..., :2] - repeat_predictions[..., :2]

    reference = output / "fly-reference.npz"
    weights = output / "legacy.weights-only.h5"
    with tempfile.TemporaryDirectory(dir=output) as staging:
        staged_reference = Path(staging) / reference.name
        staged_weights = Path(staging) / weights.name
        np.savez_compressed(
            staged_reference, frames=frames, heatmaps=heatmaps, predictions=predictions
        )
        model.train_model.save_weights(str(staged_weights), save_format="h5")

        with h5py.File(staged_weights, "r") as file:
            weights_producer = file.attrs["keras_version"]
            if isinstance(weights_producer, bytes):
                weights_producer = weights_producer.decode("utf-8")
            # HDF5 object timestamps may differ across otherwise identical exports.
            weights_digest = tensor_sha256(file)
        staged_reference.replace(reference)
        staged_weights.replace(weights)

    return {
        "source_revision": SOURCE_REVISION,
        "data_revision": DATA_REVISION,
        "asset_sha256": ASSET_HASHES,
        "output_sha256": {path.name: sha256(path) for path in (reference, weights)},
        "frame_indices": FRAME_INDICES,
        "preprocessing": "Sequential VideoReader reads with gray=True (BGR green channel).",
        "frame_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "array_shapes": {
            "frames": frames.shape,
            "heatmaps": heatmaps.shape,
            "predictions": predictions.shape,
        },
        "array_dtypes": {
            "frames": str(frames.dtype),
            "heatmaps": str(heatmaps.dtype),
            "predictions": str(predictions.dtype),
        },
        "checkpoint_keras_version": producer,
        "weights_keras_version": weights_producer,
        "weights_tensor_sha256": weights_digest,
        "pose_model": model_config,
        "training_generator": generator_config,
        "runtime": {
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "batch_size": 1,
            "tensorflow-cpu": version("tensorflow-cpu"),
            "h5py": version("h5py"),
            "numpy": version("numpy"),
            "opencv-python": version("opencv-python"),
        },
        "parity_tolerance": PARITY_TOLERANCE,
        "max_repeat_error": {
            "absolute_heatmap": float(np.max(np.abs(heatmaps - repeat_heatmaps))),
            "keypoint_euclidean_pixels": float(
                np.max(np.linalg.norm(coordinate_delta, axis=-1))
            ),
            "absolute_confidence": float(
                np.max(np.abs(predictions[..., 2] - repeat_predictions[..., 2]))
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned upstream checkout"
    )
    parser.add_argument(
        "--scoped-weights",
        action="store_true",
        help="capture repeated-build legacy weights",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="reference output directory"
    )
    parser.add_argument(
        "--backbones", nargs="+", metavar="MODEL",
        help="capture small trained backbone fixtures instead of the public fly model",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    check_source(source)
    if args.scoped_weights:
        capture_scoped_weights(source, output)
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.backbones:
        from backbone_reference import capture_backbones

        capture_backbones(source, output, args.backbones, SOURCE_REVISION)
        return
    manifest = capture(source, output)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["max_repeat_error"], sort_keys=True))


if __name__ == "__main__":
    main()
