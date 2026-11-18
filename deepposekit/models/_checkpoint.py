"""Checkpoint metadata and the legacy Keras HDF5 representation."""

import json
import re

import h5py
import numpy as np

METADATA_ENTRY = "deepposekit.json"
METADATA_FIELDS = ("pose_model_config", "train_generator_config")
HDF5_SUFFIXES = (".h5", ".hdf5")

# The original DenseNet stem contains names forbidden by Keras 3.
_DENSENET_STEM_NAMES = {
    "conv1/conv": "conv1_conv",
    "conv1/bn": "conv1_bn",
    "conv1/relu": "conv1_relu",
}

_AUTO_WEIGHTED_LAYER = re.compile(r"(conv2d|batch_normalization)(?:_(\d+))?")


def read_config(value, field):
    try:
        config = value if isinstance(value, dict) else json.loads(value)
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("class_name"), str)
            or not isinstance(config.get("config"), dict)
        ):
            raise ValueError("expected a class name and configuration")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid or missing {field} in DeepPoseKit checkpoint"
        ) from error
    return config


def normalize_legacy_graph(config):
    """Translate old Functional model and node names for Keras 3."""
    layers = config["layers"]
    nested_models = {
        layer["name"]
        for layer in layers
        if layer["class_name"] in {"Model", "Functional"}
    }
    for layer in layers:
        layer_name = layer["name"]
        layer_config = layer["config"]
        if layer_name in _DENSENET_STEM_NAMES:
            layer["name"] = layer_config["name"] = _DENSENET_STEM_NAMES[layer_name]
        if layer_name in nested_models:
            normalize_legacy_graph(layer_config)
            if layer["class_name"] == "Model":
                layer["class_name"] = "Functional"
        for node in layer["inbound_nodes"]:
            if not isinstance(node, list):
                continue
            for source in node:
                source[0] = _DENSENET_STEM_NAMES.get(source[0], source[0])
                if source[0] in nested_models:
                    # Legacy nested models reserved node zero for their internal graph.
                    if source[1] < 1:
                        raise ValueError(
                            f"Invalid legacy nested model node for {source[0]}"
                        )
                    source[1] -= 1
    for field in ("input_layers", "output_layers"):
        endpoints = config[field]
        if endpoints and isinstance(endpoints[0], str):
            endpoints = [endpoints]
        for endpoint in endpoints:
            endpoint[0] = _DENSENET_STEM_NAMES.get(endpoint[0], endpoint[0])
    if len(config["input_layers"]) == 1:
        config["input_layers"] = config["input_layers"][0]
    return config


def _hdf_names(group, attribute):
    if attribute in group.attrs:
        values = group.attrs[attribute]
    else:
        values = []
        chunk = 0
        while f"{attribute}{chunk}" in group.attrs:
            values.extend(group.attrs[f"{attribute}{chunk}"])
            chunk += 1
        if not chunk:
            raise ValueError(f"Missing {attribute} in legacy HDF5 group {group.name}")
    names = [
        value.decode("utf-8") if isinstance(value, bytes) else value for value in values
    ]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate {attribute} in legacy HDF5 group {group.name}")
    return names


def _weight_name(name):
    layer, separator, role = name.removesuffix(":0").rpartition("/")
    if not separator or not role:
        raise ValueError(f"Unsupported legacy tensor identity {name!r}")
    return f"{_DENSENET_STEM_NAMES.get(layer, layer)}/{role}"


def _legacy_weights(group):
    tensors = {}
    for name in _hdf_names(group, "layer_names"):
        layer = group.get(name)
        if not isinstance(layer, h5py.Group):
            raise ValueError(f"Missing legacy HDF5 layer {name!r}")
        for weight in _hdf_names(layer, "weight_names"):
            key = (_DENSENET_STEM_NAMES.get(name, name), _weight_name(weight))
            if key in tensors:
                raise ValueError(f"Duplicate legacy tensor identity {key}")
            dataset = layer.get(weight)
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(
                    f"Missing legacy tensor dataset {weight!r} in {name!r}"
                )
            tensors[key] = dataset
    return tensors


def _model_weights(model):
    tensors = {
        (layer.name, _weight_name(variable.path)): variable
        for layer in model.layers
        for variable in layer.weights
    }
    if len(tensors) != len(model.weights):
        raise ValueError(
            "Could not identify every model weight by layer and tensor role"
        )
    return tensors


def _relative_layer_names(tensors):
    # Keras numbers unnamed layers globally. Two independently constructed
    # networks with the same configuration can therefore differ by an offset.
    offsets = {}
    numbered = {}
    for layer, _ in tensors:
        match = _AUTO_WEIGHTED_LAYER.fullmatch(layer)
        if match:
            prefix, ordinal = match.groups()
            ordinal = int(ordinal or 0)
            numbered[layer] = (prefix, ordinal)
            offsets[prefix] = min(offsets.get(prefix, ordinal), ordinal)

    relative = {}
    for (layer, weight), tensor in tensors.items():
        if layer in numbered:
            prefix, ordinal = numbered[layer]
            normalized = f"{prefix}_{ordinal - offsets[prefix]}"
            if not weight.startswith(f"{layer}/"):
                raise ValueError(
                    f"Legacy tensor identity {weight!r} is outside layer {layer!r}"
                )
            weight = normalized + weight[len(layer) :]
            layer = normalized
        key = (layer, weight)
        if key in relative:
            raise ValueError(f"Ambiguous tensor identity {key}")
        relative[key] = tensor
    return relative


def assign_legacy_weights(model, group, *, relative_layer_names=False):
    """Validate and read every tensor before changing any model weights."""
    source = _legacy_weights(group)
    target = _model_weights(model)
    if relative_layer_names:
        source = _relative_layer_names(source)
        target = _relative_layer_names(target)
    missing = target.keys() - source.keys()
    extra = source.keys() - target.keys()
    if missing or extra:
        raise ValueError(
            "Legacy weight identities do not match the model "
            f"(missing {len(missing)}: {sorted(missing)[:3]}, "
            f"unexpected {len(extra)}: {sorted(extra)[:3]})"
        )

    assignments = []
    for key, variable in target.items():
        dataset = source[key]
        expected_dtype = np.dtype(variable.dtype)
        if tuple(variable.shape) != dataset.shape or expected_dtype != dataset.dtype:
            raise ValueError(
                f"Legacy tensor {key[1]!r} has shape {dataset.shape} and dtype "
                f"{dataset.dtype}; expected {tuple(variable.shape)} and {expected_dtype}"
            )
        assignments.append((variable, dataset[()]))

    for variable, value in assignments:
        variable.assign(value)
