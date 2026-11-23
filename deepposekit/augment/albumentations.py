"""Experimental Albumentations adapter for DeepPoseKit's ordered skeletons."""

import os
from copy import deepcopy
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from random import Random
from threading import local

import cv2
import numpy as np

ALBUMENTATIONS_VERSION = "2.0.8"
_TRANSFORMS = {
    "Affine",
    "CLAHE",
    "Defocus",
    "GaussianBlur",
    "MotionBlur",
    "OneOf",
    "Perspective",
    "Sequential",
}


def _library():
    try:
        installed = version("albumentations")
    except PackageNotFoundError as error:
        raise ValueError(
            "The experimental backend requires albumentations==2.0.8. "
            "See the README for setup with a single OpenCV wheel."
        ) from error
    if installed != ALBUMENTATIONS_VERSION:
        raise ValueError(
            f"Expected albumentations=={ALBUMENTATIONS_VERSION}, found {installed}"
        )
    for distribution in (
        "albumentationsx",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ):
        try:
            version(distribution)
        except PackageNotFoundError:
            continue
        raise ValueError(
            f"Remove {distribution} from the experiment environment. "
            "It overlaps packages already used by DeepPoseKit's Albumentations prototype."
        )
    library = import_module("albumentations")
    if library.__version__ != ALBUMENTATIONS_VERSION:
        raise ValueError(
            f"Imported Albumentations does not match {ALBUMENTATIONS_VERSION}"
        )
    return library


def _transform(library, name, options):
    from deepposekit.cli.components import named_list

    options = dict(options)
    probability = options.get("p", 0.5)
    if type(probability) not in (float, int) or not 0 <= probability <= 1:
        raise ValueError(f"{name}.p must be a probability in [0, 1]")
    if name == "FlipAxis":
        axis = options.pop("axis", 0)
        if type(axis) is not int or axis not in (0, 1):
            raise ValueError("FlipAxis axis must be 0 (vertical) or 1 (horizontal)")
        name = ("VerticalFlip", "HorizontalFlip")[axis]
    elif name not in _TRANSFORMS:
        raise ValueError(f"Unknown experimental Albumentations transform: {name}")

    if name in {"Sequential", "OneOf"}:
        children = [
            _transform(library, *spec)
            for spec in named_list(
                options.pop("transforms", None), f"{name}.transforms"
            )
        ]
        if not children:
            raise ValueError(f"{name}.transforms must contain at least one transform")
        options["transforms"] = children
    if name in {"Affine", "Perspective"} and (
        options.get("fit_output") or not options.get("keep_size", True)
    ):
        raise ValueError(f"{name} must preserve the image dimensions")
    constructor = getattr(library, name)
    try:
        signature(constructor).bind(**options)
        if issubclass(constructor, library.BasicTransform):
            options["strict"] = True
        transform = constructor(**options)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Albumentations {name} options: {error}") from error
    if name == "Affine" and transform.border_mode in {
        cv2.BORDER_REFLECT,
        cv2.BORDER_REFLECT_101,
    }:
        raise ValueError(
            "Reflective Affine borders duplicate skeleton labels in Albumentations"
        )
    return transform


class AlbumentationsAugmenter:
    """Run a recipe on image/keypoint pairs and retain a fixed skeleton layout."""

    version = ALBUMENTATIONS_VERSION

    def __init__(self, swap_index, specifications, *, random_seed=None):
        from deepposekit.cli.components import named_list

        library = _library()
        swaps = np.asarray(swap_index)
        if (
            swaps.ndim != 1
            or swaps.dtype.kind not in "iuf"
            or not np.all(
                (swaps >= -1) & (swaps < swaps.size) & (swaps == np.trunc(swaps))
            )
        ):
            raise ValueError(
                "Skeleton swap indices must be -1 or valid keypoint indices"
            )
        original = np.arange(swaps.size)
        self._swap = np.where(swaps < 0, original, swaps).astype(np.intp)
        transforms = [
            _transform(library, *spec)
            for spec in named_list(specifications, "[generator].augmenter")
        ]
        self._template = library.Compose(
            transforms,
            keypoint_params=library.KeypointParams(
                format="xy", label_fields=["keypoint_ids"], remove_invisible=False
            ),
            strict=True,
            save_applied_params=True,
        )
        self._seed = (
            int(np.random.SeedSequence().entropy)
            if random_seed is None
            else random_seed
        )
        self._local = local()

    def __getstate__(self):
        return {key: value for key, value in vars(self).items() if key != "_local"}

    def __setstate__(self, state):
        vars(self).update(state)
        self._local = local()

    def _worker(self):
        state = self._local
        pid = os.getpid()
        if getattr(state, "pid", None) != pid:
            state.pipeline = deepcopy(self._template)
            state.rng = np.random.default_rng(self._seed)
            state.pid = pid
        return state

    def _augment(self, pipeline, image, points, seed):
        indices = np.flatnonzero(
            np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1)
        )
        # Give each draw an independent stream, shared across transforms within the draw.
        pipeline.set_random_state(np.random.default_rng(seed), Random(seed))
        transformed = pipeline(
            image=image, keypoints=points[indices], keypoint_ids=indices
        )
        labels = np.asarray(transformed["keypoint_ids"], dtype=np.intp)
        if labels.size != indices.size or not np.array_equal(np.sort(labels), indices):
            raise ValueError("Albumentations changed the set of skeleton labels")
        result = np.full(points.shape, np.nan)
        result[labels] = transformed["keypoints"]
        for name, _ in transformed["applied_transforms"]:
            if name in {"HorizontalFlip", "VerticalFlip"}:
                result = result[self._swap]
        output = transformed["image"]
        if output.shape != image.shape or output.dtype != image.dtype:
            raise ValueError(
                "Albumentations must preserve the training image shape and dtype"
            )
        return output, result

    def __call__(self, *, images, keypoints, epoch=0, sample_indices=None):
        if keypoints.shape != (len(images), self._swap.size, 2):
            raise ValueError("Keypoints must match the image batch and skeleton shape")
        state = self._worker()
        if sample_indices is None:
            seeds = state.rng.integers(0, np.iinfo(np.int64).max, size=len(images))
        else:
            seeds = (
                np.random.SeedSequence((self._seed, epoch, int(index))).generate_state(
                    1, dtype=np.uint64
                )[0]
                for index in sample_indices
            )
        result = [
            self._augment(state.pipeline, image, points, int(seed))
            for image, points, seed in zip(images, keypoints, seeds, strict=True)
        ]
        if not result:
            return images, keypoints
        return tuple(np.stack(values) for values in zip(*result, strict=True))
