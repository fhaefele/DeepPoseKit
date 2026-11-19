"""Example custom reader for DeepPoseKit's predict command."""

import argparse

import numpy as np

from deepposekit.io import VideoReader

INPUT_EXTENSIONS = (".avi", ".mkv", ".mov", ".mp4")


def create_reader(path, *, batch_size, argv):
    """Return an iterator over video batches with optional preprocessing."""
    parser = argparse.ArgumentParser(prog="videoreader.py", allow_abbrev=False)
    parser.add_argument("--gray", action="store_true", help="Return one channel.")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Rotate frames counterclockwise in degrees.",
    )
    options = parser.parse_args(argv)
    return _batches(path, batch_size, options.gray, options.rotate // 90)


def _batches(path, batch_size, gray, quarter_turns):
    reader = VideoReader(str(path), batch_size=batch_size, gray=gray)
    try:
        if not reader.isOpened():
            raise OSError(f"Could not open video: {path}")
        while (frames := reader.read_batch(batch_size, asarray=True)) is not None:
            yield np.ascontiguousarray(np.rot90(frames, quarter_turns, axes=(1, 2)))
    finally:
        reader.close()
