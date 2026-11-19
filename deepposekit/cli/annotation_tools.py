"""Sample, extend, and prefill annotation datasets."""

import csv
from contextlib import ExitStack, contextmanager
from itertools import zip_longest
from math import isqrt
from os import fsdecode
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

import cv2
import h5py
import numpy as np

from deepposekit.cli.operations import (
    annotation_summary,
    atomic_output,
    check_checkpoint_skeleton,
    frame_batches,
    input_file,
    output_file,
    predict_batch,
)
from deepposekit.io import DataGenerator, initialize_dataset
from deepposekit.io.utils import initialize_skeleton
from deepposekit.models import load_model
from deepposekit.utils.keypoints import graph_to_edges


def read_skeleton(path):
    skeleton = initialize_skeleton(str(input_file(path)))
    if (
        skeleton.empty
        or skeleton["name"].isna().any()
        or not skeleton["name"].is_unique
    ):
        raise ValueError("Skeleton names must be present and unique")
    for key in ("parent", "swap"):
        unknown = set(skeleton[key].dropna()) - set(skeleton["name"])
        if unknown:
            raise ValueError(
                f"Unknown skeleton {key} name(s): {', '.join(map(str, sorted(unknown, key=str)))}"
            )
    graph_to_edges(skeleton["tree"].to_numpy())
    return skeleton


def _stride(batches, step):
    offset = 0
    for batch in batches:
        first = (-offset) % step
        if first < len(batch):
            yield offset + first, batch[first::step]
        offset += len(batch)


def _diverse_indices(args):
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(args.seed)
    limit = max(256, args.count * 4)
    candidates = []
    seen = 0
    with frame_batches(args, args.video) as batches:
        for start, batch in _stride(batches, args.frame_step):
            for index, frame in enumerate(batch):
                slot = seen if seen < limit else int(rng.integers(seen + 1))
                seen += 1
                if slot >= limit:
                    continue
                # Thumbnail features keep memory bounded independently of the input resolution.
                feature = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
                candidate = (start + index * args.frame_step, feature.ravel())
                if slot == len(candidates):
                    candidates.append(candidate)
                else:
                    candidates[slot] = candidate
    if len(candidates) < args.count:
        raise ValueError(
            f"Requested {args.count} diverse frames, but only {len(candidates)} candidates are available"
        )
    frames, features = zip(*candidates, strict=True)
    fitter = MiniBatchKMeans(
        n_clusters=min(args.count, max(2, isqrt(args.count))),
        n_init=3,
        random_state=int(rng.integers(2**32)),
    )
    labels = fitter.fit_predict(np.asarray(features, dtype=np.float32))
    groups = [
        rng.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)
    ]
    balanced = (
        frames[index]
        for row in zip_longest(*groups)
        for index in row
        if index is not None
    )
    return sorted(next(balanced) for _ in range(args.count))


def _selected_frames(batches, indices):
    requested = iter(sorted(indices))
    index = next(requested, None)
    offset = 0
    for batch in batches:
        end = offset + len(batch)
        while index is not None and index < end:
            yield index, batch[index - offset]
            index = next(requested, None)
        if index is None:
            return
        offset = end
    if index is not None:
        raise ValueError(f"Requested frame {index} is outside the source recording")


def _write_dataset(temporary, batches, skeleton):
    with ExitStack() as cleanup:
        file = None
        for samples in batches:
            if samples.dtype != np.uint8:
                raise ValueError("Annotation images must have dtype uint8")
            if file is None:
                initialize_dataset(str(temporary), samples, skeleton)
                file = cleanup.enter_context(h5py.File(temporary, "r+"))
                continue
            start = len(file["images"])
            end = start + len(samples)
            for key in ("images", "annotations", "annotated"):
                file[key].resize(end, axis=0)
            file["images"][start:end] = samples
            file["annotations"][start:end] = -1
            file["annotated"][start:end] = False
        if file is None:
            raise ValueError("The video contains no readable frames")


def dataset_init(args):
    output = output_file(
        args.output,
        ".h5",
        args.overwrite,
        args.video,
        args.skeleton,
        *([args.custom_reader] if args.custom_reader else []),
    )
    skeleton = read_skeleton(args.skeleton)
    indices = _diverse_indices(args) if args.sample == "diverse" else None
    with (
        atomic_output(output, args.overwrite) as temporary,
        frame_batches(args, args.video) as batches,
    ):
        if indices is None:
            selected = (samples for _, samples in _stride(batches, args.frame_step))
        else:
            selected = (frame[None] for _, frame in _selected_frames(batches, indices))
        _write_dataset(temporary, selected, skeleton)
    print(output)


def _annotation_output(path, overwrite, *inputs):
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("Annotation outputs must end in .h5 or .hdf5")
    return output_file(path, path.suffix, overwrite, *inputs)


def _frame_indices(path):
    with input_file(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or reader.fieldnames.count("frame") != 1:
            raise ValueError("Frame CSV must have a single 'frame' column")
        indices = {}
        for row in reader:
            try:
                index = int(row["frame"])
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid frame index on CSV line {reader.line_num}"
                ) from error
            if index < 0 or index in indices:
                raise ValueError(
                    f"Negative or duplicate frame index on CSV line {reader.line_num}: {index}"
                )
            indices[index] = len(indices)
    if not indices:
        raise ValueError("Frame CSV contains no candidate frames")
    return indices


def _reference_dtype(dtype):
    if h5py.check_dtype(ref=dtype) is not None:
        return True
    if dtype.fields is not None:
        return any(_reference_dtype(field[0]) for field in dtype.fields.values())
    if dtype.subdtype is not None:
        return _reference_dtype(dtype.subdtype[0])
    base = h5py.check_dtype(vlen=dtype)
    return isinstance(base, np.dtype) and _reference_dtype(base)


def _contains_references(obj):
    return (isinstance(obj, h5py.Dataset) and _reference_dtype(obj.dtype)) or any(
        _reference_dtype(obj.attrs.get_id(name).dtype) for name in obj.attrs
    )


def _rewrite_needed(data, count):
    return data.chunks is None or (
        data.maxshape[0] is not None and count > data.maxshape[0]
    )


def _protect_references(file, arrays, count):
    replacements = [array for array in arrays if _rewrite_needed(array, count)]
    if not replacements:
        return
    if (
        any(h5py.h5o.get_info(array.id).rc > 1 for array in replacements)
        or _contains_references(file)
        or file.visititems(
            lambda _name, obj: True if _contains_references(obj) else None
        )
    ):
        raise ValueError(
            "HDF5 object references or hard links prevent extending fixed annotation arrays"
        )


def _external_links(file):
    links = {}

    def collect(name, link):
        if isinstance(link, h5py.ExternalLink):
            links[name] = link

    file.visititems_links(collect)
    return links


def _external_dependencies(source):
    source = input_file(source).absolute()
    dependencies = set()

    def add(name):
        path = Path(fsdecode(name))
        if path != Path("."):
            dependencies.update(
                (source.parent / path, source.resolve().parent / path, path.absolute())
            )

    def collect_storage(_name, obj):
        if isinstance(obj, h5py.Dataset):
            for filename, _, _ in obj.external or ():
                add(filename)
            if obj.is_virtual:
                for mapping in obj.virtual_sources():
                    add(mapping.file_name)

    with h5py.File(source, "r") as file:
        for link in _external_links(file).values():
            add(link.filename)
        file.visititems(collect_storage)
        for key in ("skeleton", "skeleton_names"):
            data = file.get(key)
            if isinstance(data, h5py.Dataset):
                if data.file.id != file.id:
                    add(data.file.filename)
                collect_storage(key, data)
    return dependencies


@contextmanager
def _annotation_copy(source, destination, dataset):
    copyfile(source, destination)
    with h5py.File(source, "r") as original, h5py.File(destination, "r+") as output:
        _local_arrays(original, dataset)
        _local_arrays(output, dataset)
        # Resolve external skeleton objects relative to the source before relocation.
        for key in ("skeleton", "skeleton_names"):
            if key not in original:
                continue
            data = original[key]
            if not isinstance(data, h5py.Dataset):
                continue
            if (
                data.file.id == original.id
                and not data.external
                and not data.is_virtual
            ):
                continue
            if _contains_references(data):
                raise ValueError(
                    f"Cannot copy external {key} metadata with HDF5 object references"
                )
            del output[key]
            inline = output.create_dataset(key, data=data[:], dtype=data.dtype)
            for name, value in data.attrs.items():
                inline.attrs[name] = value
        for name, link in _external_links(output).items():
            backing = Path(link.filename)
            if not backing.is_absolute():
                del output[name]
                output[name] = h5py.ExternalLink(
                    str(source.absolute().parent / backing), link.path
                )
        yield output


def _local_arrays(file, dataset):
    arrays = tuple(file[key] for key in (dataset, "annotations", "annotated"))
    for array in arrays:
        if array.file.id != file.id or array.external or array.is_virtual:
            raise ValueError(
                f"Cannot update annotation arrays backed by external or virtual storage: {array.name}"
            )
    return arrays


def _extend(file, key, count):
    data = file[key]
    if not _rewrite_needed(data, count):
        data.resize(count, axis=0)
        return
    # Older HDF5 inputs can have fixed-size arrays. Extend the staged copy in chunks.
    temporary_name = f"_deepposekit_{uuid4().hex}"
    replacement = file.create_dataset(
        temporary_name,
        shape=(count, *data.shape[1:]),
        dtype=data.dtype,
        maxshape=(None, *data.shape[1:]),
        chunks=data.chunks,
        compression=data.compression,
        compression_opts=data.compression_opts,
        shuffle=data.shuffle,
        fletcher32=data.fletcher32,
        scaleoffset=data.scaleoffset,
        fillvalue=data.fillvalue,
    )
    for name, value in data.attrs.items():
        replacement.attrs[name] = value
    step = max(1, min(256, 8_388_608 // max(1, data[0:1].nbytes)))
    for start in range(0, len(data), step):
        end = min(start + step, len(data))
        replacement[start:end] = data[start:end]
    del file[key]
    file.move(temporary_name, key)


def append(args):
    inputs = [args.annotations, args.source, args.frames]
    if args.custom_reader is not None:
        inputs.append(args.custom_reader)
    dependencies = _external_dependencies(args.annotations)
    output = _annotation_output(args.output, args.overwrite, *inputs, *dependencies)
    annotation_summary(args.annotations, args.dataset)
    indices = _frame_indices(args.frames)
    with atomic_output(output, args.overwrite) as temporary:
        with (
            _annotation_copy(args.annotations, temporary, args.dataset) as file,
            frame_batches(args, args.source) as batches,
        ):
            start = len(file[args.dataset])
            end = start + len(indices)
            _protect_references(file, _local_arrays(file, args.dataset), end)
            for key in (args.dataset, "annotations", "annotated"):
                _extend(file, key, end)
            images = file[args.dataset]
            file["annotations"][start:] = -1
            file["annotated"][start:] = False
            for index, frame in _selected_frames(batches, indices):
                if frame.shape != images.shape[1:] or frame.dtype != images.dtype:
                    raise ValueError(
                        f"Frame {index} has shape {frame.shape} and dtype {frame.dtype}. "
                        f"Expected shape {images.shape[1:]} and dtype {images.dtype}."
                    )
                images[start + indices[index]] = frame
    print(output)


def prelabel(args):
    dependencies = _external_dependencies(args.annotations)
    output = _annotation_output(
        args.output, args.overwrite, args.annotations, args.model, *dependencies
    )
    annotation_summary(args.annotations, args.dataset)
    data = DataGenerator(str(args.annotations), dataset=args.dataset, mode="full")
    model = load_model(args.model, compile=False)
    check_checkpoint_skeleton(model, data)
    if data.image_shape != tuple(model.predict_model.input_shape[1:]):
        raise ValueError("Annotation image shape does not match the checkpoint")
    if data.keypoints_shape[0] != model.predict_model.output_shape[1]:
        raise ValueError("Checkpoint and annotations have different keypoint counts")
    with atomic_output(output, args.overwrite) as temporary:
        with _annotation_copy(args.annotations, temporary, args.dataset) as file:
            images, points, reviewed = _local_arrays(file, args.dataset)
            for start in range(0, len(images), args.batch_size):
                end = start + args.batch_size
                missing = ~reviewed[start:end].astype(bool)
                rows = np.any(missing, axis=1)
                if not rows.any():
                    continue
                predicted = predict_batch(
                    model,
                    images[start:end][rows],
                    start // args.batch_size + 1,
                    args.batch_size,
                )
                if predicted.shape != (np.count_nonzero(rows), points.shape[1], 3):
                    raise ValueError(
                        "Checkpoint and annotations have different keypoint counts"
                    )
                keypoints = points[start:end]
                proposals = predicted[..., :2][missing[rows]]
                if not np.isfinite(proposals).all():
                    raise ValueError("Model returned non-finite keypoint coordinates")
                keypoints[missing] = proposals
                points[start:end] = keypoints
    print(output)
