"""Run the CLI reference examples with a small generated video and annotations."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib

import cv2
import h5py
import numpy as np

from deepposekit.cli.examples import EXAMPLES

TRAINING = """[data]
annotations = "labeled.h5"

[generator]
downsample_factor = 0
use_graph = false
random_seed = 17
augmenter = [{ name = "FlipAxis", axis = 1, p = 0.5 }]

[model]
name = "LEAP"
filters = 4
subpixel = false

[compile]
jit_compile = false

[training]
epochs = 1
batch_size = 2
steps_per_epoch = 1
verbose = 2

[output]
checkpoint = "model.keras"
"""


def _write_annotations(path, images, labels):
    with h5py.File(path, "x") as file:
        file["images"] = images
        file["annotations"] = labels
        file["annotated"] = np.ones(labels.shape[:2], dtype=bool)
        file["skeleton"] = np.array([[-1, 1], [0, 0]], dtype=np.int32)
        file["skeleton_names"] = np.array([b"head", b"tail"])


def _prepare(directory):
    images = np.random.default_rng(17).integers(0, 255, (8, 32, 32, 1), dtype=np.uint8)
    labels = np.repeat([[[7.0, 10.0], [20.0, 22.0]]], len(images), axis=0)
    writer = cv2.VideoWriter(
        str(directory / "clip.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 5, (32, 32)
    )
    try:
        if not writer.isOpened():
            raise RuntimeError("Could not create the sample AVI with OpenCV")
        for image in images:
            writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    _write_annotations(directory / "labeled.h5", images[:6], labels[:6])
    _write_annotations(directory / "holdout.h5", images[6:], labels[6:])
    (directory / "skeleton.csv").write_text(
        "name,parent,swap\nhead,,tail\ntail,head,head\n", encoding="utf-8"
    )
    (directory / "frames.csv").write_text("frame\n0\n3\n", encoding="utf-8")
    (directory / "training.toml").write_text(TRAINING, encoding="utf-8")


def _check_files(directory):
    predictions = np.load(directory / "predictions.npy", allow_pickle=False)
    np.testing.assert_equal(predictions.shape, (8, 2, 3))
    if not np.isfinite(predictions).all():
        raise RuntimeError("The prediction example returned non-finite keypoints")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if not np.isfinite(metrics["mean_error_px"]):
        raise RuntimeError("The evaluation example returned a non-finite error")
    for name, count in (
        ("unlabeled.h5", 8),
        ("expanded.h5", 8),
        ("prelabeled.h5", 8),
        ("combined.h5", 14),
        ("subset.h5", 4),
        ("parts/part-003.h5", 6),
    ):
        with h5py.File(directory / name) as file:
            np.testing.assert_equal(len(file["images"]), count)
    with h5py.File(directory / "prelabeled.h5") as file:
        if np.all(file["annotations"][:] == -1) or file["annotated"][:].any():
            raise RuntimeError(
                "Prelabeling must add suggestions and preserve review flags"
            )
    for name in ("preview.png", "trace.png", "overlay.avi", "candidates.csv"):
        if not (directory / name).stat().st_size:
            raise RuntimeError(f"The examples did not write {name}")


def _run(directory, require_gpu):
    _prepare(directory)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    for name in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        env.setdefault(name, "1")
    if require_gpu:
        env["TF_CPP_MIN_LOG_LEVEL"] = "0"
    for _, argv in EXAMPLES:
        print(shlex.join(["deepposekit", *argv]), flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "deepposekit", *argv],
            cwd=directory,
            env=env,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"Example exited with {result.returncode}\n{result.stdout}{result.stderr}"
            )
        if argv == ("doctor",):
            report = json.loads(result.stdout)
            print("TensorFlow devices:", ", ".join(report["devices"]), flush=True)
            if require_gpu and not report["gpu_available"]:
                raise RuntimeError(
                    "TensorFlow did not report a GPU. Check deepposekit doctor."
                )
        elif argv[:2] == ("imgaug", "toml"):
            names = [
                item["name"]
                for item in tomllib.loads(result.stdout)["generator"]["augmenter"]
            ]
            if names != list(argv[2:]):
                raise RuntimeError("The augmentation example changed transform order")
        elif argv[0] == "help":
            if json.loads(result.stdout)["command"]["path"] != ["dataset", "split"]:
                raise RuntimeError("JSON help returned the wrong command")
        elif require_gpu and argv[0] in {"train", "predict"}:
            print(result.stderr, end="", file=sys.stderr, flush=True)
            if "device:GPU:" not in result.stderr:
                raise RuntimeError(
                    f"{argv[0]} did not initialize a TensorFlow GPU device"
                )
    _check_files(directory)
    print(f"Passed {len(EXAMPLES)} CLI examples.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require a visible GPU and verify GPU initialization during training and prediction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep the generated inputs and results in a new directory.",
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        directory = args.output_dir.resolve()
        directory.mkdir()
        context = nullcontext(directory)
    else:
        context = TemporaryDirectory(prefix="deepposekit-cli-")
    with context as directory:
        _run(Path(directory), args.require_gpu)


if __name__ == "__main__":
    main()
