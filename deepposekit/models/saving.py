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

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import h5py
import keras

from deepposekit.models._checkpoint import HDF5_SUFFIXES, METADATA_ENTRY
from deepposekit.utils.io import get_json_type


def _model_metadata(model):
    generator = model.train_generator
    if generator is None:
        metadata = getattr(model, "_checkpoint_metadata", None)
        if metadata is None:
            raise ValueError("DeepPoseKit geometry is required to save this model")
        return metadata

    return {
        "pose_model_config": {
            "class_name": type(model).__name__,
            "config": model.get_config(),
        },
        "train_generator_config": {
            "class_name": type(generator).__name__,
            "config": generator.get_config(),
        },
    }


def save_model(model, path, optimizer=True, overwrite=True):
    """Save a DeepPoseKit model and its inference metadata.

    Use a `.keras` filename for the modern format or `.h5` / `.hdf5` for HDF5.
    Inference-only models imported from a checkpoint retain their original geometry.
    """
    path = Path(path)
    if path.suffix not in (*HDF5_SUFFIXES, ".keras"):
        raise ValueError("Save DeepPoseKit checkpoints as .keras, .h5, or .hdf5")
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    metadata = _model_metadata(model)
    network = model.train_model
    if not optimizer:
        network = keras.Model(
            inputs=network.input,
            outputs=network.output,
            name=network.name,
            trainable=network.trainable,
        )
    with TemporaryDirectory(dir=path.parent, prefix=".deepposekit-") as directory:
        temporary_path = Path(directory) / path.name
        keras.saving.save_model(network, temporary_path)
        if path.suffix == ".keras":
            with ZipFile(temporary_path, "a") as archive:
                archive.writestr(
                    METADATA_ENTRY, json.dumps(metadata, default=get_json_type)
                )
                for name, contents in getattr(model, "_training_files", {}).items():
                    archive.writestr(name, contents)
        else:
            with h5py.File(temporary_path, "r+") as checkpoint:
                for name, config in metadata.items():
                    checkpoint.attrs[name] = json.dumps(config, default=get_json_type)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
