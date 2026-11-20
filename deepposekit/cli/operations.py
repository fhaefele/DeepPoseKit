"""Shared command-line data preparation, inference, and output handling."""

from contextlib import ExitStack, contextmanager
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
from shutil import copyfileobj
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from unicodedata import normalize
from uuid import uuid4

import h5py
import numpy as np

from deepposekit.io import DataGenerator, VideoReader
from deepposekit.models import load_model, load_legacy_weights
from deepposekit.models.loading import read_checkpoint_metadata
from deepposekit.utils.keypoints import graph_to_edges, keypoint_errors


def input_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _portable_path_key(path):
    # Outputs may be on a case-insensitive volume, even on POSIX.
    return normalize("NFC", os.path.normcase(path).casefold())


def _check_output_file(path, suffix, overwrite, protected):
    if path.suffix != suffix:
        raise ValueError(f"Output must end in {suffix}: {path}")
    if path.is_dir():
        raise ValueError(f"Output needs to be a file: {path}")
    if path.resolve() in protected:
        raise ValueError(f"Output would replace an input: {path}")
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )


def output_file(path, suffix, overwrite, *inputs):
    path = Path(path)
    _check_output_file(
        path,
        suffix,
        overwrite,
        {Path(source).resolve() for source in inputs},
    )
    if not path.parent.is_dir():
        raise ValueError(f"Output needs an existing parent directory: {path}")
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


def annotation_arrays(file, dataset, *, allow_empty=False):
    names = (dataset, "annotations", "annotated", "skeleton")
    if not all(isinstance(file.get(name), h5py.Dataset) for name in names):
        raise ValueError(
            f"Expected a DeepPoseKit annotation HDF5 file with image key {dataset!r}"
        )
    images, points, labeled, skeleton = (file[name] for name in names)
    if (
        images.ndim != 4
        or min(images.shape[1:]) < 1
        or (not allow_empty and not len(images))
    ):
        description = "images" if allow_empty else "nonempty images"
        raise ValueError(
            f"Expected {description} shaped (frames, height, width, channels)"
        )
    if skeleton.ndim != 2 or skeleton.shape[1] != 2 or not len(skeleton):
        raise ValueError("Expected a nonempty skeleton with parent and swap columns")
    expected = (len(images), len(skeleton))
    if points.shape != (*expected, 2) or labeled.shape != expected:
        raise ValueError("Annotation dimensions do not match the images and skeleton")
    graph_to_edges(skeleton[:, 0])
    swap = skeleton[:, 1]
    if not np.all(
        np.isfinite(swap)
        & (swap == np.floor(swap))
        & (swap >= -1)
        & (swap < len(skeleton))
    ):
        raise ValueError("Skeleton swap indices must be -1 or valid keypoint indices")
    return images, points, labeled, skeleton


def annotation_summary(path, dataset):
    with h5py.File(input_file(path), "r") as file:
        images, _, labeled, skeleton = annotation_arrays(file, dataset)
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


@contextmanager
def custom_module(path, kind):
    path = input_file(path)
    if path.suffix != ".py":
        raise ValueError(f"A custom {kind} must be a Python file (.py)")
    name = f"_deepposekit_{kind}_{uuid4().hex}"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load custom {kind}: {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Load the explicitly selected source without creating a bytecode file
        # in the recording directory.
        exec(
            compile(path.read_bytes(), spec.origin, "exec", dont_inherit=True),
            vars(module),
        )
        yield module
    finally:
        sys.modules.pop(name, None)


@contextmanager
def custom_reader_module(path):
    if path is None:
        yield None
        return
    with custom_module(path, "reader") as module:
        if not callable(getattr(module, "create_reader", None)):
            raise ValueError(
                "Custom reader must define create_reader(path, *, batch_size, argv)"
            )
        yield module


def _prediction_streams(module, argv):
    if module is None:
        return {None: None}
    plan = getattr(module, "prediction_streams", None)
    if plan is None:

        def open_recording(path, *, batch_size):
            return module.create_reader(path, batch_size=batch_size, argv=list(argv))

        return {None: open_recording}
    streams = plan(list(argv))
    if (
        not isinstance(streams, dict)
        or not streams
        or not all(
            isinstance(name, str) and callable(opener)
            for name, opener in streams.items()
        )
    ):
        raise ValueError("prediction_streams(argv) must return named reader factories")
    return streams


@contextmanager
def prediction_reader(args, path, opener):
    if opener is None:
        with open_video(path, args.batch_size, args.gray) as reader:
            yield reader.iter_batches()
        return
    reader = opener(path, batch_size=args.batch_size)
    with ExitStack() as cleanup:
        close_reader = getattr(reader, "close", None)
        if callable(close_reader):
            cleanup.callback(close_reader)
        batches = iter(reader)
        close_batches = getattr(batches, "close", None)
        if batches is not reader and callable(close_batches):
            cleanup.callback(close_batches)
        yield batches


def _validated_batches(batches, size):
    shape = None
    for number, batch in enumerate(batches, 1):
        if (
            not isinstance(batch, np.ndarray)
            or batch.ndim != 4
            or not 1 <= len(batch) <= size
            or min(batch.shape[1:]) < 1
            or batch.dtype.kind not in "uif"
        ):
            raise ValueError(
                f"Reader batch {number} must contain up to {size} numeric images"
            )
        if shape is not None and batch.shape[1:] != shape:
            raise ValueError(f"Reader batch {number} changed the frame shape")
        shape = batch.shape[1:]
        yield batch


@contextmanager
def frame_batches(args, source):
    """Use the prediction reader contract to consume one recording stream."""
    source = input_file(source)
    with custom_reader_module(args.custom_reader) as module:
        streams = _prediction_streams(module, args.custom_reader_args or [])
        if len(streams) != 1:
            raise ValueError("Select exactly one recording stream for this command")
        with prediction_reader(args, source, next(iter(streams.values()))) as batches:
            yield _validated_batches(batches, args.batch_size)


VIDEO_EXTENSIONS = frozenset(
    ".avi .mp4 .mov .mkv .webm .mpeg .mpg .m4v .wmv .mts .m2ts .ts .asf .flv .ogv .3gp".split()
)


def _directory_extensions(module):
    if module is None:
        return VIDEO_EXTENSIONS
    extensions = getattr(module, "INPUT_EXTENSIONS", None)
    if extensions is None:
        return None
    if (
        not isinstance(extensions, (list, tuple, set, frozenset))
        or not all(
            isinstance(suffix, str) and re.fullmatch(r"\.[A-Za-z0-9]+", suffix)
            for suffix in extensions
        )
        or not extensions
    ):
        raise ValueError("INPUT_EXTENSIONS must contain file extensions such as '.h5'")
    return {suffix.lower() for suffix in extensions}


def _raise_scan_error(error):
    raise error


def _scan_recordings(root, output_root, recursive, extensions, protected):
    for directory, subdirs, filenames in root.walk(on_error=_raise_scan_error):
        subdirs[:] = (
            sorted(
                name
                for name in subdirs
                if name != "__pycache__" and (directory / name).resolve() != output_root
            )
            if recursive
            else []
        )
        for name in sorted(filenames):
            path = directory / name
            if (
                (extensions is None or path.suffix.lower() in extensions)
                and path.is_file()
                and path.resolve() not in protected
            ):
                yield path, path.relative_to(root)


def _file_identity(path):
    try:
        details = path.stat()
    except FileNotFoundError:
        return None
    if details.st_ino:
        return details.st_dev, details.st_ino
    return None


def _prediction_jobs(args, module, streams):
    protected = {args.checkpoint.resolve()}
    if args.custom_reader is not None:
        protected.add(args.custom_reader.resolve())
    if args.output_dir is None:
        if len(streams) != 1:
            raise ValueError("Use --output-dir when predicting multiple reader streams")
        source, destination = args.paths
        if source.is_dir():
            raise ValueError("Use --output-dir when predicting a directory")
        output_file(destination, ".npy", args.overwrite, source, *protected)
        name, opener = next(iter(streams.items()))
        return [(source, destination, name, opener)]

    output_root = args.output_dir.resolve()
    sources = []
    unfiltered_roots = []
    for path in args.paths:
        if path.is_dir():
            extensions = _directory_extensions(module)
            root = path.resolve()
            if (extensions is None or ".npy" in extensions) and (
                output_root == root or not output_root.is_relative_to(root)
            ):
                unfiltered_roots.append(root)
            found = list(
                _scan_recordings(
                    path,
                    output_root,
                    args.recursive,
                    extensions,
                    protected,
                )
            )
            if not found:
                raise ValueError(f"No matching recordings in {path}")
            sources.extend(found)
        else:
            sources.append((input_file(path), Path(path.name)))

    protected.update(source.resolve() for source, _ in sources)
    protected_files = (
        {_file_identity(path) for path in protected} if args.overwrite else set()
    )
    protected_files.discard(None)
    for name in streams:
        if name is not None and (
            not name
            or name.endswith((".", " "))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        ):
            raise ValueError(
                f"Reader stream name cannot be used in a filename: {name!r}"
            )

    jobs = []
    destinations = {}
    for source, relative in sources:
        for name, opener in streams.items():
            filename = (
                f"{relative.stem}.{name}.npy"
                if name is not None
                else f"{relative.stem}.npy"
            )
            destination = args.output_dir / relative.with_name(filename)
            parent = destination.parent.resolve()
            if not parent.is_relative_to(output_root):
                raise ValueError(
                    f"Output parent points outside --output-dir: {destination.parent}"
                )
            for ancestor in (destination.parent, *destination.parent.parents):
                if ancestor.is_dir():
                    break
                if os.path.lexists(ancestor):
                    raise ValueError(f"Output parent is not a directory: {ancestor}")
            output_path = parent / filename
            if any(output_path.is_relative_to(root) for root in unfiltered_roots):
                raise ValueError(
                    "Use a separate --output-dir when the reader's directory scan includes .npy files"
                )
            key = _portable_path_key(output_path)
            if key in destinations:
                raise ValueError(
                    f"Output collision for {destinations[key]} and {source}: {destination}"
                )
            _check_output_file(destination, ".npy", args.overwrite, protected)
            if args.overwrite and _file_identity(destination) in protected_files:
                raise ValueError(f"Output would replace an input: {destination}")
            destinations[key] = source
            jobs.append((source, destination, name, opener))
    for _, destination, _, _ in jobs:
        parent = destination.parent.resolve()
        for ancestor in (parent, *parent.parents):
            if _portable_path_key(ancestor) in destinations:
                raise ValueError(
                    f"Output path is also needed as a directory: {ancestor}"
                )
    return jobs


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


def _predict_recording(args, source, output, opener, model):
    count = 0
    result_shape = None
    with TemporaryFile(dir=output.parent) as spool:
        with prediction_reader(args, source, opener) as batches:
            input_file(source)
            if model is None:
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
    return model


def predict(args):
    model = None
    with custom_reader_module(args.custom_reader) as module:
        streams = _prediction_streams(module, args.custom_reader_args or [])
        for source, output, name, opener in _prediction_jobs(args, module, streams):
            if args.output_dir is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
            try:
                model = _predict_recording(args, source, output, opener, model)
            except Exception as error:
                if args.output_dir is None:
                    raise
                label = str(source) if name is None else f"{source} ({name})"
                raise RuntimeError(f"{label}: {error}") from error
            print(output)


def convert(args):
    source = input_file(args.input)
    if not h5py.is_hdf5(source):
        raise ValueError("Expected a legacy HDF5 checkpoint or weights file")
    protected = (source,) if args.config is None else (source, args.config)
    output = output_file(args.output, ".keras", args.overwrite, *protected)
    with ExitStack() as resources:
        if args.config is None:
            model = load_model(source, compile=False)
        else:
            from deepposekit.cli.data_source import open_data
            from deepposekit.cli.training import build_model, read_training_config

            config = read_training_config(args.config, for_weights=True)
            output_file(output, ".keras", args.overwrite, *config.data.inputs)
            data = resources.enter_context(open_data(config.data))
            model = build_model(config, data, initialize_weights=False)
            load_legacy_weights(model, source)
        model.save(output, overwrite=args.overwrite)
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


def check_checkpoint_skeleton(model, data, *, label="Annotations"):
    stored = model._checkpoint_metadata["train_generator_config"]["config"]
    for name, actual in (
        ("skeleton", np.column_stack((data.graph, data.swap_index)).tolist()),
        ("skeleton_names", data.skeleton_names),
    ):
        if name in stored and stored[name] != actual:
            raise ValueError(f"{label} {name} does not match the checkpoint")


def evaluate(args):
    if args.output is not None:
        output_file(
            args.output, ".json", args.overwrite, args.checkpoint, args.annotations
        )
    annotation_summary(args.annotations, args.dataset)
    data = DataGenerator(str(args.annotations), dataset=args.dataset)
    model = load_model(args.checkpoint, compile=False)
    check_checkpoint_skeleton(model, data, label="Holdout")
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


def doctor(args):
    from importlib.metadata import PackageNotFoundError, version

    import keras
    import tensorflow as tf

    from _deepposekit_gpu import bundled_cuda_installed
    from deepposekit import __version__

    try:
        cuda_runtime = version("nvidia-cuda-runtime-cu12")
    except PackageNotFoundError:
        cuda_runtime = None
    devices = tf.config.list_physical_devices()
    print(
        json.dumps(
            {
                "command": args.command_name,
                "version": args.command_version,
                "deepposekit": __version__,
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "cuda_wheels_installed": bundled_cuda_installed(),
                "cuda_runtime": cuda_runtime,
                "gpu_available": any(device.device_type == "GPU" for device in devices),
                "devices": [device.name for device in devices],
            },
            indent=2,
        )
    )


def run(args):
    if args.command == "train":
        from deepposekit.cli.training import train

        train(args)
    elif args.command == "doctor":
        doctor(args)
    elif args.command == "dataset":
        if args.dataset_command in {"merge", "split"}:
            from deepposekit.cli import dataset_tools

            {"merge": dataset_tools.merge, "split": dataset_tools.split}[
                args.dataset_command
            ](args)
            return
        from deepposekit.cli import annotation_tools

        {
            "init": annotation_tools.dataset_init,
            "append": annotation_tools.append,
            "prelabel": annotation_tools.prelabel,
        }[args.dataset_command](args)
    elif args.command == "predictions":
        from deepposekit.cli import prediction_tools

        prediction_tools.run(args)
    else:
        commands = {
            "predict": predict,
            "convert": convert,
            "inspect": inspect_input,
            "evaluate": evaluate,
        }
        commands[args.command](args)
