"""Combine or extract rows from initialized DeepPoseKit annotation sets."""

from pathlib import Path
import os
import sys
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


def _open_annotations(file, dataset, *, allow_empty=False):
    images, points, reviewed, skeleton = annotation_arrays(
        file, dataset, allow_empty=allow_empty
    )
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
    for name in source.attrs:
        attribute = source.attrs.get_id(name)
        dtype = attribute.dtype
        if _reference_dtype(dtype):
            continue
        # Preserve the file's HDF5 type and dataspace. The NumPy buffer uses
        # h5py's memory type so variable-length values can be copied safely.
        values = None
        if attribute.shape is not None:
            values = np.zeros(attribute.shape, dtype=dtype)
            memory_type = h5py.h5t.py_create(dtype)
            attribute.read(values, mtype=memory_type)
        encoded_name = name.encode("utf-8") if isinstance(name, str) else name
        copied = h5py.h5a.create(
            target.id, encoded_name, attribute.get_type().copy(), attribute.get_space()
        )
        try:
            if values is not None:
                copied.write(values, mtype=memory_type)
        finally:
            copied.close()


def _attribute_values_equal(first, second):
    if isinstance(first, h5py.Empty) or isinstance(second, h5py.Empty):
        return isinstance(first, h5py.Empty) and isinstance(second, h5py.Empty)
    left, right = np.asarray(first), np.asarray(second)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.fields is not None:
        return all(
            _attribute_values_equal(left[field], right[field])
            for field in left.dtype.names
        )
    if left.dtype.kind == "O":
        return all(
            _attribute_values_equal(a, b)
            for a, b in zip(left.flat, right.flat, strict=True)
        )
    if left.dtype.kind == "c":
        return np.array_equal(left.real, right.real, equal_nan=True) and np.array_equal(
            left.imag, right.imag, equal_nan=True
        )
    return np.array_equal(left, right, equal_nan=left.dtype.kind == "f")


def _equal_attributes(first, second, attr):
    left = first.get_id(attr)
    right = second.get_id(attr)
    return (
        left.shape == right.shape
        and left.get_type() == right.get_type()
        and _attribute_values_equal(first[attr], second[attr])
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
            dtype=h5py.Datatype(array.id.get_type().copy()),
            chunks=True,
            track_order=True,
        )
        _copy_attributes(array, output)
        destination[key] = output
    for key, array in metadata.items():
        output = file.create_dataset(
            key,
            data=array[:],
            dtype=h5py.Datatype(array.id.get_type().copy()),
            track_order=True,
        )
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
    for key, (shape, h5type) in template.items():
        if arrays[key].shape[1:] != shape or arrays[key].id.get_type() != h5type:
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
        h5py.File(temporary, "w", track_order=True) as result,
    ):
        size = 0
        for index, source in enumerate(sources):
            with h5py.File(source, "r") as file:
                arrays, metadata, names = _open_annotations(
                    file, args.dataset, allow_empty=True
                )
                rows = len(arrays[args.dataset])
                if index == 0:
                    destination = _create_output(result, file, arrays, metadata, rows)
                    template = {
                        key: (array.shape[1:], array.id.get_type().copy())
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
    with h5py.File(path, "x", track_order=True) as result:
        destination = _create_output(result, file, arrays, metadata, size)
        _copy_rows(arrays, destination, ranges)


def _publish_directory(stage, output):
    """Publish the completed directory without making an empty destination visible."""
    if os.name == "nt":
        stage.rename(output)
        return

    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        at_fdcwd = -100
        rename_noreplace = 1
        libc.renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        code = libc.renameat2(
            at_fdcwd,
            os.fsencode(stage),
            at_fdcwd,
            os.fsencode(output),
            rename_noreplace,
        )
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename_excl = 4
        libc.renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        code = libc.renamex_np(os.fsencode(stage), os.fsencode(output), rename_excl)
    else:
        code = -1
        ctypes.set_errno(errno.ENOSYS)

    if code != 0:
        error = ctypes.get_errno()
        if error in (errno.ENOSYS, errno.ENOTSUP, errno.EINVAL):
            raise OSError(
                "This filesystem does not support publishing a new partition directory atomically"
            )
        raise OSError(error, os.strerror(error), output)


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
