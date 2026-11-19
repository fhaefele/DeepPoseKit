"""Shared command-line data preparation, inference, and output handling."""

from contextlib import ExitStack, contextmanager
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from uuid import uuid4

import h5py
import numpy as np

from deepposekit.io import DataGenerator, VideoReader, initialize_dataset
from deepposekit.io.utils import initialize_skeleton
from deepposekit.models import load_model, load_legacy_weights
from deepposekit.models.loading import read_checkpoint_metadata
from deepposekit.utils.keypoints import graph_to_edges, keypoint_errors


def input_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def output_file(path, suffix, overwrite, *inputs):
    path = Path(path)
    if path.suffix != suffix:
        raise ValueError(f"Output must end in {suffix}: {path}")
    if not path.parent.is_dir() or path.is_dir():
        raise ValueError(f"Output needs an existing parent directory: {path}")
    if any(path.resolve() == Path(source).resolve() for source in inputs):
        raise ValueError(f"Output would replace an input: {path}")
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )
    return path


@contextmanager
def atomic_output(path, overwrite):
    """Publish a complete artifact on the same filesystem."""
    with TemporaryDirectory(dir=path.parent, prefix=".deepposekit-") as directory:
        temporary = Path(directory) / path.name
        yield temporary
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)


def annotation_summary(path, dataset):
    with h5py.File(input_file(path), "r") as file:
        names = (dataset, "annotations", "annotated", "skeleton")
        if not all(isinstance(file.get(name), h5py.Dataset) for name in names):
            raise ValueError(
                f"Expected a DeepPoseKit annotation HDF5 file with image key {dataset!r}"
            )
        images, points, labeled, skeleton = (file[name] for name in names)
        if images.ndim != 4 or min(images.shape, default=0) < 1:
            raise ValueError(
                "Expected nonempty images shaped (frames, height, width, channels)"
            )
        if skeleton.ndim != 2 or skeleton.shape[1] != 2 or not len(skeleton):
            raise ValueError(
                "Expected a nonempty skeleton with parent and swap columns"
            )
        expected = (len(images), len(skeleton))
        if points.shape != (*expected, 2) or labeled.shape != expected:
            raise ValueError(
                "Annotation dimensions do not match the images and skeleton"
            )
        graph_to_edges(skeleton[:, 0])
        swap = skeleton[:, 1]
        if not np.all(
            np.isfinite(swap)
            & (swap == np.floor(swap))
            & (swap >= -1)
            & (swap < len(skeleton))
        ):
            raise ValueError(
                "Skeleton swap indices must be -1 or valid keypoint indices"
            )
        return {
            "kind": "annotations",
            "dataset": dataset,
            "frames": len(images),
            "labeled_frames": int(np.count_nonzero(np.all(labeled[:], axis=1))),
            "image_shape": list(images.shape[1:]),
            "keypoints": len(skeleton),
        }


@contextmanager
def open_video(path, batch_size, gray):
    reader = VideoReader(str(input_file(path)), batch_size=batch_size, gray=gray)
    try:
        if not reader.isOpened():
            raise ValueError(f"Could not open video: {path}")
        yield reader
    finally:
        reader.close()


def video_batches(reader, batch_size):
    while (batch := reader.read_batch(batch_size, asarray=True)) is not None:
        yield batch


@contextmanager
def prediction_reader(args):
    if args.custom_reader is None:
        with open_video(args.input, args.batch_size, args.gray) as reader:
            yield video_batches(reader, args.batch_size)
        return
    path = input_file(args.custom_reader)
    if path.suffix != ".py":
        raise ValueError("A custom reader must be a Python file (.py)")
    name = f"_deepposekit_reader_{uuid4().hex}"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load custom reader: {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, "create_reader", None)
        if not callable(factory):
            raise ValueError(
                "Custom reader must define create_reader(path, *, batch_size, argv)"
            )
        reader = factory(
            args.input, batch_size=args.batch_size, argv=args.custom_reader_args or []
        )
        with ExitStack() as cleanup:
            close_reader = getattr(reader, "close", None)
            if callable(close_reader):
                cleanup.callback(close_reader)
            batches = iter(reader)
            close_batches = getattr(batches, "close", None)
            if batches is not reader and callable(close_batches):
                cleanup.callback(close_batches)
            yield batches
    finally:
        sys.modules.pop(name, None)


def predict_batch(model, frames, number, batch_size):
    shape = tuple(model.predict_model.input_shape[1:])
    if (
        not isinstance(frames, np.ndarray)
        or frames.ndim != 4
        or not 1 <= len(frames) <= batch_size
        or frames.shape[1:] != shape
        or frames.dtype.kind not in "uif"
    ):
        actual = getattr(frames, "shape", None)
        raise ValueError(
            f"Batch {number} has invalid images {actual}. Expected up to {batch_size} frames of shape {shape}."
        )
    return np.asarray(model.predict_on_batch(frames), dtype=np.float32)


def predict(args):
    inputs = [args.checkpoint, args.input]
    if args.custom_reader is not None:
        inputs.append(args.custom_reader)
    output = output_file(args.output, ".npy", args.overwrite, *inputs)
    count = 0
    result_shape = None
    with TemporaryFile(dir=output.parent) as spool:
        with prediction_reader(args) as batches:
            input_file(args.input)
            model = load_model(args.checkpoint, compile=False)
            for number, frames in enumerate(batches, 1):
                predicted = predict_batch(model, frames, number, args.batch_size)
                if (
                    predicted.ndim != 3
                    or predicted.shape[0] != len(frames)
                    or predicted.shape[2] != 3
                ):
                    raise ValueError(
                        f"Batch {number} returned invalid predictions: {predicted.shape}"
                    )
                if result_shape is not None and predicted.shape[1:] != result_shape:
                    raise ValueError(
                        f"Batch {number} returned an inconsistent keypoint shape"
                    )
                result_shape = predicted.shape[1:]
                spool.write(predicted.tobytes())
                count += len(predicted)
        if not count:
            raise ValueError("The recording contains no readable frames")
        with atomic_output(output, args.overwrite) as temporary:
            with temporary.open("wb") as result:
                np.lib.format.write_array_header_2_0(
                    result,
                    {
                        "descr": np.dtype(np.float32).str,
                        "fortran_order": False,
                        "shape": (count, *result_shape),
                    },
                )
                spool.seek(0)
                copyfileobj(spool, result)
    print(output)


def convert(args):
    source = input_file(args.input)
    if not h5py.is_hdf5(source):
        raise ValueError("Expected a legacy HDF5 checkpoint or weights file")
    protected = (source,) if args.config is None else (source, args.config)
    output = output_file(args.output, ".keras", args.overwrite, *protected)
    if args.config is None:
        model = load_model(source, compile=False)
    else:
        from deepposekit.cli.training import build_model, read_training_config

        config = read_training_config(args.config, for_weights=True)
        output_file(output, ".keras", args.overwrite, config.annotations)
        model = build_model(config)
        load_legacy_weights(model, source)
    model.save(output, overwrite=args.overwrite)
    print(output)


def dataset_init(args):
    output = output_file(args.output, ".h5", args.overwrite, args.video, args.skeleton)
    skeleton = initialize_skeleton(str(input_file(args.skeleton)))
    if (
        skeleton.empty
        or skeleton["name"].isna().any()
        or not skeleton["name"].is_unique
    ):
        raise ValueError("Skeleton names must be present and unique")
    graph_to_edges(skeleton["tree"].to_numpy())
    samples = []
    with open_video(args.video, 8, args.gray) as reader:
        index = 0
        for batch in video_batches(reader, 8):
            samples.extend(batch[(-index) % args.frame_step :: args.frame_step])
            index += len(batch)
    if not samples:
        raise ValueError("The video contains no readable frames")
    with atomic_output(output, args.overwrite) as temporary:
        initialize_dataset(str(temporary), np.stack(samples), skeleton)
    print(output)


def inspect_input(args):
    path = input_file(args.input)
    kind = "checkpoint"
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            if "annotations" in file or "annotated" in file:
                kind = "annotations"
    if kind == "annotations":
        result = annotation_summary(path, args.dataset)
    else:
        metadata = read_checkpoint_metadata(path)["pose_model_config"]
        result = {
            "kind": kind,
            "model": metadata["class_name"],
            **{
                key: metadata["config"][key]
                for key in (
                    "image_shape",
                    "keypoints_shape",
                    "downsample_factor",
                    "subpixel",
                )
            },
        }
    print(json.dumps(result, indent=2, allow_nan=False))


def evaluate(args):
    if args.output is not None:
        output_file(
            args.output, ".json", args.overwrite, args.checkpoint, args.annotations
        )
    annotation_summary(args.annotations, args.dataset)
    data = DataGenerator(str(args.annotations), dataset=args.dataset)
    model = load_model(args.checkpoint, compile=False)
    stored = model._checkpoint_metadata["train_generator_config"]["config"]
    for name, actual in (
        ("skeleton", np.column_stack((data.graph, data.swap_index)).tolist()),
        ("skeleton_names", data.skeleton_names),
    ):
        if name in stored and stored[name] != actual:
            raise ValueError(f"Holdout {name} does not match the checkpoint")
    errors = []
    for start in range(0, len(data), args.batch_size):
        indices = range(start, min(start + args.batch_size, len(data)))
        predicted = predict_batch(
            model, data.get_images(indices), len(errors) + 1, args.batch_size
        )[..., :2]
        actual = data.get_keypoints(indices)
        if actual.shape != predicted.shape:
            raise ValueError("Checkpoint and holdout have different keypoint counts")
        if not np.isfinite(predicted).all():
            raise ValueError("Model returned non-finite keypoint coordinates")
        valid = np.all(np.isfinite(actual) & (actual >= 0), axis=-1)
        errors.append(np.where(valid, keypoint_errors(actual, predicted)[1], np.nan))
    errors = np.concatenate(errors)
    valid = errors[np.isfinite(errors)]
    if not valid.size:
        raise ValueError("No valid labeled keypoints to evaluate")
    result = {
        "frames": len(data),
        "mean_error_px": float(np.mean(valid)),
        "median_error_px": float(np.median(valid)),
        "per_keypoint_mean_px": [
            float(np.mean(values[np.isfinite(values)]))
            if np.isfinite(values).any()
            else None
            for values in errors.T
        ],
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        with atomic_output(args.output, args.overwrite) as temporary:
            temporary.write_text(serialized, encoding="utf-8")
        print(args.output)


def doctor():
    import keras
    import tensorflow as tf

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "devices": [
                    device.name for device in tf.config.list_physical_devices()
                ],
            },
            indent=2,
        )
    )


def run(args):
    if args.command == "train":
        from deepposekit.cli.training import train

        train(args)
    elif args.command == "doctor":
        doctor()
    else:
        commands = {
            "predict": predict,
            "convert": convert,
            "dataset": dataset_init,
            "inspect": inspect_input,
            "evaluate": evaluate,
        }
        commands[args.command](args)
