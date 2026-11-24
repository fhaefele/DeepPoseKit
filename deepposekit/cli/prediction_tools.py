"""Review saved DeepPoseKit pose predictions without loading a model."""

import csv
from contextlib import closing
from heapq import nlargest
from itertools import chain
from tempfile import TemporaryFile

import cv2
import h5py
import numpy as np

from deepposekit.cli.annotation_tools import read_skeleton
from deepposekit.cli.operations import (
    atomic_output,
    frame_batches,
    input_file,
    output_file,
)
from deepposekit.cli.render import display_or_save, require_display, skeleton_edges
from deepposekit.io.DataGenerator import read_skeleton_names


def _load_predictions(path):
    path = input_file(path)
    if path.suffix != ".npy":
        raise ValueError("Expected a prediction .npy file")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(values, np.ndarray):
        values.close()
        raise ValueError("Expected a prediction .npy array")
    if (
        values.ndim != 3
        or min(values.shape[:2], default=0) < 1
        or values.shape[2] != 3
        or values.dtype.kind not in "fui"
    ):
        raise ValueError("Expected numeric predictions shaped (frames, keypoints, 3)")
    return values


def _skeleton(path, keypoints):
    if path is None:
        return (), None
    if path.suffix.lower() == ".csv":
        data = read_skeleton(path)
        graph, names = data["tree"].to_numpy(), data["name"].tolist()
    else:
        with h5py.File(input_file(path), "r") as file:
            skeleton = file.get("skeleton")
            if (
                not isinstance(skeleton, h5py.Dataset)
                or skeleton.ndim != 2
                or skeleton.shape[1] != 2
            ):
                raise ValueError(
                    "Expected annotation skeleton with parent and swap columns"
                )
            graph = skeleton[:, 0]
            names = read_skeleton_names(file, len(graph))
    if len(graph) != keypoints:
        raise ValueError("Skeleton and prediction keypoint counts differ")
    return skeleton_edges(graph), names


def _range(args, size):
    stop = size if args.stop is None else args.stop
    if not 0 <= args.start < stop <= size:
        raise ValueError(f"Expected 0 <= --start < --stop <= {size}")
    return args.start, stop


def _keypoint(value, total, names, explicit_name):
    if explicit_name is not None:
        if names is None or explicit_name not in names:
            raise ValueError(
                f"Unknown keypoint name {explicit_name!r}. Provide a matching --skeleton."
            )
        return names.index(explicit_name)
    try:
        index = int(value)
    except ValueError:
        index = None
    if index is not None and str(index) == value and 0 <= index < total:
        return index
    if names is not None and value in names:
        return names.index(value)
    if index is None:
        raise ValueError(
            f"Unknown keypoint {value!r}. Provide a zero-based index or a matching --skeleton."
        )
    if not 0 <= index < total:
        raise ValueError(f"Keypoint index must be between 0 and {total - 1}")
    return index


def _output(args, allowed, *inputs):
    if args.output is None:
        require_display()
        return None
    if args.output.suffix not in allowed:
        raise ValueError(f"Output must end in {', '.join(allowed)}")
    return output_file(
        args.output,
        args.output.suffix,
        args.overwrite,
        *(path for path in inputs if path is not None),
    )


def trace(args):
    predictions = _load_predictions(args.predictions)
    start, stop = _range(args, len(predictions))
    _, names = _skeleton(args.skeleton, predictions.shape[1])
    total = predictions.shape[1]
    if args.keypoint_name:
        selected = (_keypoint(None, total, names, name) for name in args.keypoint_name)
    else:
        selected = (
            _keypoint(value, total, names, None) for value in args.keypoint or ["0"]
        )
    labels = {
        index: str(index) if names is None else names[index] for index in selected
    }
    output = _output(args, (".png",), args.predictions, args.skeleton)
    from matplotlib import pyplot as plt

    frames = np.arange(start, stop)
    threshold = args.confidence_threshold
    figure, grid = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    x_axis, y_axis, confidence_axis, trajectory = grid.flat
    time_axes = (x_axis, y_axis, confidence_axis)
    for axis, label in zip(time_axes, ("X (px)", "Y (px)", "Confidence"), strict=True):
        axis.set(xlabel="Frame", ylabel=label)

    if len(labels) <= 20:
        palette = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
        colors = palette.colors[: len(labels)]
    else:
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(labels)))

    for (index, label), color in zip(labels.items(), colors, strict=True):
        x, y, confidence = predictions[start:stop, index].T
        x_axis.plot(frames, x, color=color, label=label)
        y_axis.plot(frames, y, color=color)
        confidence_axis.plot(frames, confidence, color=color)
        if threshold is not None:
            keep = np.isfinite(confidence) & (confidence >= threshold)
            # Frame centers are integers. Shade each full frame, including isolated hits.
            edges = np.flatnonzero(np.diff(keep, prepend=False, append=False))
            starts, stops = edges[::2], edges[1::2]
            spans = np.column_stack((starts + start - 0.5, stops - starts))
            if spans.size:
                for axis in time_axes:
                    axis.broken_barh(
                        spans,
                        (0, 1),
                        transform=axis.get_xaxis_transform(),
                        facecolors=color,
                        alpha=0.15,
                        linewidth=0,
                    )
            x, y = np.where(keep, (x, y), np.nan)
        trajectory.plot(x, y, color=color, marker="o", markersize=3, linewidth=1)

    if threshold is not None:
        confidence_axis.axhline(threshold, color="0.4", linestyle="--", linewidth=1)
    if len(labels) == 1:
        figure.suptitle(f"Keypoint {next(iter(labels.values()))}")
    else:
        figure.suptitle("Keypoint traces")
        x_axis.legend()
    trajectory.set(xlabel="X (px)", ylabel="Y (px)", title="Image trajectory")
    trajectory.set_aspect("equal", adjustable="datalim")
    trajectory.invert_yaxis()
    display_or_save(figure, output, args.overwrite)


def _as_bgr(frame):
    if frame.dtype != np.uint8:
        if frame.dtype.kind == "u" and frame.dtype.itemsize > 1:
            frame = frame.astype(np.float32) * (255.0 / np.iinfo(frame.dtype).max)
        elif (
            frame.dtype.kind == "f"
            and np.isfinite(frame).all()
            and 0 <= frame.min() <= frame.max() <= 1
        ):
            frame = frame * 255
        frame = np.clip(np.nan_to_num(frame), 0, 255).astype(np.uint8)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] != 3:
        raise ValueError(f"Cannot display frames with {frame.shape[2]} channels")
    return np.array(frame, order="C", copy=True)


def _draw(frame, prediction, edges):
    canvas = _as_bgr(frame)
    height, width = canvas.shape[:2]
    points = prediction[:, :2]
    valid = np.all(
        np.isfinite(points) & (points >= 0) & (points < (width, height)), axis=1
    )
    visible = {
        index: tuple(map(int, np.rint(points[index])))
        for index in np.flatnonzero(valid)
    }
    for first, second in edges:
        if first in visible and second in visible:
            cv2.line(
                canvas, visible[first], visible[second], (0, 215, 255), 1, cv2.LINE_AA
            )
    for point in visible.values():
        cv2.circle(canvas, point, 3, (0, 255, 40), -1, cv2.LINE_AA)
    return canvas


def _overlay_frames(args, predictions, edges, start, stop):
    count = 0
    with frame_batches(args, args.source) as batches:
        for batch in batches:
            end = count + len(batch)
            if end > len(predictions):
                raise ValueError(
                    f"The source has more than {len(predictions)} prediction frames"
                )
            for index in range(max(count, start), min(end, stop)):
                yield _draw(batch[index - count], predictions[index], edges)
            count = end
    if count != len(predictions):
        raise ValueError(
            f"The source has {count} frames for {len(predictions)} prediction frames"
        )


def _export_view(path, frames, args, count):
    with atomic_output(path, args.overwrite) as temporary:
        first = next(frames, None)
        if first is None:
            raise ValueError("The source contains no frames in this range")
        height, width = first.shape[:2]
        size = (width + width % 2, height + height % 2)
        codec = "FFV1" if path.suffix == ".avi" else "mp4v"
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*codec), args.fps, size
        )
        try:
            if not writer.isOpened():
                raise OSError(f"Could not open {path.suffix} video output")

            def write(frame):
                writer.write(
                    cv2.copyMakeBorder(
                        frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT
                    )
                )

            write(first)
            for frame in frames:
                write(frame)
        finally:
            writer.release()
        check = cv2.VideoCapture(str(temporary))
        try:
            if (
                not check.isOpened()
                or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != count
            ):
                raise OSError(
                    "The exported video does not contain the requested frame range"
                )
        finally:
            check.release()
    print(path)


def _scrub(frames, start, stop):
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Slider

    with TemporaryFile() as spool:
        first = next(frames, None)
        if first is None:
            raise ValueError("The source contains no frames in this range")
        locations = []
        for frame in chain((first,), frames):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError("Could not stage a frame for the interactive viewer")
            locations.append((spool.tell(), encoded.nbytes))
            spool.write(encoded)
        figure, axis = plt.subplots()
        axis.set_axis_off()
        figure.subplots_adjust(bottom=0.15)
        image = axis.imshow(first[..., ::-1])
        axis.set_title(f"Frame {start}")
        slider = None
        if stop - start > 1:
            slider = Slider(
                figure.add_axes([0.15, 0.05, 0.7, 0.04]),
                "Frame",
                start,
                stop - 1,
                valinit=start,
                valstep=1,
            )

            def update(value):
                index = int(value)
                offset, size = locations[index - start]
                spool.seek(offset)
                encoded = np.frombuffer(spool.read(size), dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                image.set_data(frame[..., ::-1])
                axis.set_title(f"Frame {index}")
                figure.canvas.draw_idle()

            slider.on_changed(update)
        display_or_save(figure, None, False)


def view(args):
    predictions = _load_predictions(args.predictions)
    start, stop = _range(args, len(predictions))
    edges, _ = _skeleton(args.skeleton, predictions.shape[1])
    output = _output(
        args,
        (".avi", ".mp4"),
        args.predictions,
        args.source,
        args.skeleton,
        args.custom_reader,
    )
    with closing(_overlay_frames(args, predictions, edges, start, stop)) as frames:
        if output is None:
            _scrub(frames, start, stop)
        else:
            _export_view(output, frames, args, stop - start)


def _outlier_rows(predictions, min_drop, min_jump):
    # Work in chunks so a long recording does not need a second full prediction array.
    for start in range(1, len(predictions), 8192):
        end = min(start + 8192, len(predictions))
        delta = np.diff(
            np.asarray(predictions[start - 1 : end], dtype=np.float64), axis=0
        )
        drop = np.max(
            np.where(np.isfinite(delta[..., 2]), -delta[..., 2], 0), axis=1, initial=0
        )
        jump = np.hypot(delta[..., 0], delta[..., 1])
        jump = np.max(np.where(np.isfinite(jump), jump, 0), axis=1, initial=0)
        for offset in np.flatnonzero((drop >= min_drop) | (jump >= min_jump)):
            signals = []
            if drop[offset] >= min_drop:
                signals.append("confidence_drop")
            if jump[offset] >= min_jump:
                signals.append("jump")
            score = max(drop[offset] / min_drop, jump[offset] / min_jump)
            yield (
                start + int(offset),
                score,
                drop[offset],
                jump[offset],
                "+".join(signals),
            )


def outliers(args):
    predictions = _load_predictions(args.predictions)
    output = output_file(args.output, ".csv", args.overwrite, args.predictions)
    rows = nlargest(
        args.top,
        _outlier_rows(predictions, args.min_drop, args.min_jump),
        key=lambda row: (row[1], -row[0]),
    )
    with atomic_output(output, args.overwrite) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("frame", "score", "confidence_drop", "jump_px", "signals"))
            writer.writerows(rows)
    print(output)


def run(args):
    {"view": view, "trace": trace, "outliers": outliers}[args.predictions_command](args)
