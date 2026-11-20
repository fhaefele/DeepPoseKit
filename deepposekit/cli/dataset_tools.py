"""Combine or extract rows from initialized DeepPoseKit annotation sets."""

from pathlib import Path
import os
import re
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from deepposekit.cli.annotation_tools import (
    _annotation_output,
    _external_dependencies,
    _reference_dtype,
)
from deepposekit.cli.operations import annotation_arrays, atomic_output, input_file
from deepposekit.io.DataGenerator import read_skeleton_names


def _open_annotations(file, dataset):
    images, points, reviewed, skeleton = annotation_arrays(file, dataset)
    arrays = {dataset: images, "annotations": points, "annotated": reviewed}
    for key, array in arrays.items():
        allowed = "biu" if key == "annotated" else "iuf"
        if array.dtype.kind not in allowed:
            raise ValueError(
                f"{array.name} has an unsupported annotation dtype: {array.dtype}"
            )
    names = read_skeleton_names(file, len(skeleton))
    if "skeleton_names" in file and names is None:
        raise ValueError(f"Invalid or ambiguous skeleton names in {file.filename}")
    metadata = {"skeleton": skeleton}
    if names is not None:
        metadata["skeleton_names"] = file["skeleton_names"]
    return arrays, metadata, names


def _copy_attributes(source, target):
    for attr in source.attrs:
        dtype = source.attrs.get_id(attr).dtype
        if not _reference_dtype(dtype):
            target.attrs.create(attr, source.attrs[attr], dtype=dtype)


def _equal_attributes(first, second, attr):
    left = first.get_id(attr).dtype
    right = second.get_id(attr).dtype
    return left == right and np.array_equal(
        first[attr], second[attr], equal_nan=left.kind in "fc"
    )


def _keep_matching_attributes(target, source):
    for attr in list(target.attrs):
        if attr not in source.attrs or not _equal_attributes(
            target.attrs, source.attrs, attr
        ):
            del target.attrs[attr]


def _create_output(file, template, arrays, metadata, rows):
    _copy_attributes(template, file)
    destination = {}
    for key, array in arrays.items():
        output = file.create_dataset(
            key,
            shape=(rows, *array.shape[1:]),
            maxshape=(None, *array.shape[1:]),
            dtype=array.dtype,
            chunks=True,
        )
        _copy_attributes(array, output)
        destination[key] = output
    for key, array in metadata.items():
        output = file.create_dataset(key, data=array[:], dtype=array.dtype)
        _copy_attributes(array, output)
    return destination


def _copy_rows(arrays, destination, ranges, offset=0):
    for start, stop in ranges:
        length = stop - start
        for key, source in arrays.items():
            bytes_per_row = source.dtype.itemsize * int(np.prod(source.shape[1:]))
            batch = max(1, 16 * 1024 * 1024 // bytes_per_row)
            for step in range(0, length, batch):
                size = min(batch, length - step)
                destination[key][offset + step : offset + step + size] = source[
                    start + step : start + step + size
                ]
        offset += length


def _check_compatible(
    template, skeleton, names, file, arrays, source_skeleton, source_names
):
    for key, (shape, dtype) in template.items():
        if arrays[key].shape[1:] != shape or arrays[key].dtype != dtype:
            raise ValueError(
                f"{file.filename} has incompatible {key} dimensions or dtype"
            )
    if not np.array_equal(skeleton, source_skeleton):
        raise ValueError(
            f"{file.filename} has a different skeleton graph or swap order"
        )
    if names != source_names:
        raise ValueError(
            f"{file.filename} has different skeleton names or keypoint order"
        )


def merge(args):
    if len(args.annotations) < 2:
        raise ValueError("Pass at least two annotation files to dataset merge")
    sources = [input_file(path) for path in args.annotations]
    dependencies = {
        path for source in sources for path in _external_dependencies(source)
    }
    output = _annotation_output(args.output, args.overwrite, *sources, *dependencies)
    with (
        atomic_output(output, args.overwrite) as temporary,
        h5py.File(temporary, "w") as result,
    ):
        size = 0
        for index, source in enumerate(sources):
            with h5py.File(source, "r") as file:
                arrays, metadata, names = _open_annotations(file, args.dataset)
                rows = len(arrays[args.dataset])
                if index == 0:
                    destination = _create_output(result, file, arrays, metadata, rows)
                    template = {
                        key: (array.shape[1:], array.dtype)
                        for key, array in arrays.items()
                    }
                    skeleton = metadata["skeleton"][:]
                    template_names = names
                else:
                    _check_compatible(
                        template,
                        skeleton,
                        template_names,
                        file,
                        arrays,
                        metadata["skeleton"][:],
                        names,
                    )
                    _keep_matching_attributes(result, file)
                    for key, target in destination.items():
                        _keep_matching_attributes(target, arrays[key])
                        target.resize(size + rows, axis=0)
                    for key, array in metadata.items():
                        _keep_matching_attributes(result[key], array)
                _copy_rows(arrays, destination, [(0, rows)], size)
                size += rows
    print(output)


def _frame_ranges(value, count):
    ranges = []
    for token in value.split(","):
        match = re.fullmatch(r"([0-9]+)(?::([0-9]+))?", token.strip())
        if match is None:
            raise ValueError(
                f"Invalid frame selector {token!r}. Use an index or START:STOP."
            )
        start = int(match[1])
        stop = start + 1 if match[2] is None else int(match[2])
        if not 0 <= start < stop <= count:
            raise ValueError(
                f"Frame selector {token!r} is outside the {count} annotation rows or is empty"
            )
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], stop)
        else:
            ranges.append((start, stop))
    return ranges


def _partition_ranges(value, count):
    tokens = [token.strip() for token in value.split(",")]
    if not all(re.fullmatch(r"[0-9]+", token) for token in tokens):
        raise ValueError("Split points must be comma-separated row indices")
    points = (0, *map(int, tokens), count)
    ranges = list(zip(points, points[1:]))
    if not all(start < stop for start, stop in ranges):
        raise ValueError(
            f"Split points must be strictly increasing and inside the {count} annotation rows"
        )
    return ranges


def _write_subset(path, file, arrays, metadata, ranges):
    size = sum(stop - start for start, stop in ranges)
    with h5py.File(path, "x") as result:
        destination = _create_output(result, file, arrays, metadata, size)
        _copy_rows(arrays, destination, ranges)


def _publish_directory(stage, output):
    if os.name == "nt":
        # On Windows rename fails if the destination exists.
        stage.rename(output)
        return
    output.mkdir()
    reserved = output.stat(follow_symlinks=False)
    try:
        stage.replace(output)
    except BaseException:
        try:
            if os.path.samestat(reserved, output.stat(follow_symlinks=False)):
                output.rmdir()
        except OSError:
            pass
        raise


def split(args):
    source = input_file(args.annotations)
    if args.at is None:
        if args.output is None:
            raise ValueError("--frames requires --output")
        output = _annotation_output(
            args.output, args.overwrite, source, *_external_dependencies(source)
        )
    else:
        if args.output_dir is None or args.overwrite:
            raise ValueError("--at requires a new --output-dir")
        output = args.output_dir
        if os.path.lexists(output):
            raise FileExistsError(f"Output directory already exists: {output}")
        if not output.parent.is_dir():
            raise ValueError(f"Output directory needs an existing parent: {output}")
        root = output.resolve()
        if any(
            root in dependency.resolve().parents
            for dependency in _external_dependencies(source)
        ):
            raise ValueError(
                f"Output directory contains an input backing path: {output}"
            )

    with h5py.File(source, "r") as file:
        arrays, metadata, _ = _open_annotations(file, args.dataset)
        rows = len(arrays[args.dataset])
        if args.frames is not None:
            ranges = _frame_ranges(args.frames, rows)
            with atomic_output(output, args.overwrite) as temporary:
                _write_subset(temporary, file, arrays, metadata, ranges)
            print(output)
            return

        ranges = _partition_ranges(args.at, rows)
        width = max(3, len(str(len(ranges))))
        with TemporaryDirectory(dir=output.parent, prefix=".deepposekit-") as temporary:
            stage = Path(temporary) / "parts"
            stage.mkdir()
            for index, interval in enumerate(ranges, 1):
                name = f"part-{index:0{width}d}.h5"
                _write_subset(stage / name, file, arrays, metadata, [interval])
            _publish_directory(stage, output)
        for index in range(1, len(ranges) + 1):
            print(output / f"part-{index:0{width}d}.h5")
