# -*- coding: utf-8 -*-
# Copyright 2018-2019 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
from importlib import import_module
from math import isfinite
from numbers import Integral
from pathlib import Path
from zipfile import ZipFile

import h5py
import keras

from deepposekit.io import TrainingGenerator
from deepposekit.models._checkpoint import (
    METADATA_ENTRY,
    METADATA_FIELDS,
    assign_legacy_weights,
    normalize_legacy_graph,
    read_config,
)
from deepposekit.models.layers.convolutional import (
    SubPixelDownscaling,
    SubPixelUpscaling,
    UpSampling2D,
)
from deepposekit.models.layers.deeplabcut import ImageNetPreprocess
from deepposekit.models.layers.util import ImageNormalization

_MODELS = {"LEAP", "StackedDenseNet", "StackedHourglass", "DeepLabCut"}
_CUSTOM_LAYERS = {
    cls.__name__: cls
    for cls in (
        ImageNormalization,
        UpSampling2D,
        SubPixelDownscaling,
        SubPixelUpscaling,
        ImageNetPreprocess,
    )
}
_MODEL_GEOMETRY = (
    "image_shape",
    "output_shape",
    "keypoints_shape",
    "n_output_channels",
    "downsample_factor",
    "output_sigma",
)
_SHAPE_FIELDS = {"image_shape", "output_shape", "keypoints_shape"}


def _read_metadata(values):
    if not isinstance(values, (dict, h5py.AttributeManager)):
        raise ValueError("Invalid DeepPoseKit checkpoint metadata")
    return {field: read_config(values.get(field), field) for field in METADATA_FIELDS}


def _legacy_checkpoint_layout(checkpoint):
    if "model_weights" in checkpoint or any(
        field in checkpoint.attrs for field in ("model_config", *METADATA_FIELDS)
    ):
        if "model_config" not in checkpoint.attrs:
            raise ValueError("Missing model_config in full HDF5 checkpoint")
        if not isinstance(checkpoint.get("model_weights"), h5py.Group):
            raise ValueError(
                "Missing or invalid model_weights group in full HDF5 checkpoint"
            )
        return "full"
    if any(name in checkpoint.attrs for name in ("layer_names", "layer_names0")):
        return "weights-only"
    raise ValueError(
        "Unsupported HDF5 checkpoint layout. Expected a DeepPoseKit checkpoint "
        "or a legacy Keras weights file with root layer_names."
    )


def _load_legacy(path, custom_objects, compile_model):
    with h5py.File(path, "r") as checkpoint:
        if _legacy_checkpoint_layout(checkpoint) == "weights-only":
            raise ValueError(
                "Expected a full DeepPoseKit checkpoint. For weights-only HDF5 use "
                "load_legacy_weights with a matching StackedDenseNet model."
            )
        metadata = _read_metadata(checkpoint.attrs)
        graph = read_config(checkpoint.attrs["model_config"], "model_config")
        if graph["class_name"] not in {"Model", "Functional"}:
            raise ValueError(
                f"Unsupported legacy Keras model type {graph['class_name']!r}"
            )
        try:
            graph_config = normalize_legacy_graph(graph["config"])
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Invalid legacy Keras model graph") from error
        model = keras.Model.from_config(graph_config, custom_objects=custom_objects)
        assign_legacy_weights(model, checkpoint["model_weights"])
        training_config = checkpoint.attrs.get("training_config")

    if compile_model and training_config is not None:
        spec = json.loads(training_config)
        optimizer = spec["optimizer_config"]
        options = optimizer["config"]
        if "lr" in options:
            options.setdefault("learning_rate", options.pop("lr"))
        options.pop("decay", None)
        model.compile(
            optimizer=keras.optimizers.deserialize(
                optimizer, custom_objects=custom_objects
            ),
            loss=spec["loss"],
            metrics=spec.get("metrics") or None,
            weighted_metrics=spec.get("weighted_metrics") or None,
            loss_weights=spec.get("loss_weights"),
        )
    return model, metadata


def _load_modern(path, custom_objects, compile_model):
    with ZipFile(path) as archive:
        try:
            with archive.open(METADATA_ENTRY) as entry:
                metadata = _read_metadata(json.load(entry))
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"{path} lacks valid DeepPoseKit checkpoint metadata"
            ) from error
    model = keras.saving.load_model(
        path, custom_objects=custom_objects, compile=compile_model
    )
    return model, metadata


def _geometry(config):
    try:
        geometry = {
            key: tuple(config[key]) if key in _SHAPE_FIELDS else config[key]
            for key in _MODEL_GEOMETRY
        }
        image = geometry["image_shape"]
        output = geometry["output_shape"]
        keypoints = geometry["keypoints_shape"]
        channels = geometry["n_output_channels"]
        downsample = geometry["downsample_factor"]
        sigma = geometry["output_sigma"]
        if (
            len(image) != 3
            or len(output) != 2
            or len(keypoints) != 2
            or not all(
                isinstance(size, Integral) and not isinstance(size, bool) and size > 0
                for size in (*image, *output, *keypoints, channels)
            )
            or keypoints[1] != 2
            or keypoints[0] > channels
            or not isinstance(downsample, Integral)
            or isinstance(downsample, bool)
            or downsample < 0
            or isinstance(sigma, bool)
            or not isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("invalid inference dimensions or decoder settings")
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError(
            "Missing or invalid DeepPoseKit model geometry in checkpoint"
        ) from error
    if tuple(size >> downsample for size in image[:2]) != output:
        raise ValueError(
            f"DeepPoseKit downsample_factor {downsample} is inconsistent with "
            f"image shape {image[:2]} and heatmap shape {output}"
        )
    return geometry


def _restore_model(train_model, metadata, generator, augmenter):
    pose = metadata["pose_model_config"]
    config = pose["config"]
    name = pose["class_name"]
    if name not in _MODELS:
        raise ValueError(f"Unsupported DeepPoseKit model type {name!r}")
    if not isinstance(config.get("subpixel"), bool):
        raise ValueError(
            "DeepPoseKit pose_model_config requires a boolean subpixel setting"
        )
    model_type = getattr(import_module(f"deepposekit.models.{name}"), name)
    geometry = _geometry(config)
    saved_generator = metadata["train_generator_config"]["config"]
    if _geometry(saved_generator) != geometry:
        raise ValueError("DeepPoseKit pose and training-generator metadata disagree")
    if tuple(train_model.inputs[0].shape[1:]) != geometry["image_shape"] or tuple(
        train_model.outputs[-1].shape[1:]
    ) != (*geometry["output_shape"], geometry["n_output_channels"]):
        raise ValueError(
            "DeepPoseKit model geometry disagrees with the stored Keras graph"
        )

    train_generator = None
    if generator is not None:
        params = inspect.signature(TrainingGenerator).parameters
        options = {
            key: saved_generator[key]
            for key in params.keys() & saved_generator.keys()
            if key not in {"generator", "augmenter"}
        }
        train_generator = TrainingGenerator(
            generator=generator, augmenter=augmenter, **options
        )
        if _geometry(train_generator.get_config()) != geometry:
            raise ValueError(
                "The supplied training generator does not match the checkpoint geometry"
            )

    params = inspect.signature(model_type).parameters
    model_options = {
        key: config[key]
        for key in params.keys() & config.keys()
        if key != "train_generator"
    }
    model = model_type(train_generator, skip_init=True, **model_options)
    model.train_model = train_model
    model._checkpoint_metadata = metadata
    model.__init_train_model__()
    model.__init_input__(geometry["image_shape"])
    model.__init_predict_model__(
        geometry["output_shape"],
        geometry["keypoints_shape"],
        geometry["downsample_factor"],
        geometry["output_sigma"],
    )
    return model


def load_model(path, generator=None, augmenter=None, custom_objects=None, compile=True):
    """Load a DeepPoseKit checkpoint from legacy HDF5 or modern Keras format.

    Supply a compatible annotation generator to train the returned model.
    With `compile=False`, the model is ready for inference or manual compilation.
    A legacy checkpoint loaded with `compile=True` starts with a fresh optimizer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    objects = {**_CUSTOM_LAYERS, **(custom_objects or {})}
    if h5py.is_hdf5(path):
        train_model, metadata = _load_legacy(path, objects, compile)
    elif path.suffix == ".keras":
        train_model, metadata = _load_modern(path, objects, compile)
    else:
        raise ValueError(f"Unsupported DeepPoseKit checkpoint format at {path}")
    return _restore_model(train_model, metadata, generator, augmenter)


def load_legacy_weights(model, path):
    """Import legacy Keras HDF5 weights into an initialized StackedDenseNet."""
    from deepposekit.models.StackedDenseNet import StackedDenseNet

    if not isinstance(model, StackedDenseNet) or not isinstance(
        model.train_model, keras.Model
    ):
        raise TypeError(
            "Pass an initialized StackedDenseNet matching the legacy weights"
        )
    with h5py.File(path, "r") as checkpoint:
        if _legacy_checkpoint_layout(checkpoint) == "full":
            raise ValueError("Use load_model for full DeepPoseKit checkpoints")
        assign_legacy_weights(model.train_model, checkpoint, existing_model=True)
    return model
