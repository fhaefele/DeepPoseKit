"""Checkpoint metadata and the legacy Keras HDF5 representation."""

import json
import re

import h5py
import keras
import numpy as np

METADATA_ENTRY = "deepposekit.json"
TRAINING_CONFIG_ENTRY = "training.toml"
TRAINING_SETTINGS_ENTRY = "training-run.json"
TRAINING_ENTRIES = (TRAINING_CONFIG_ENTRY, TRAINING_SETTINGS_ENTRY)
METADATA_FIELDS = ("pose_model_config", "train_generator_config")
HDF5_SUFFIXES = (".h5", ".hdf5")

# The original DenseNet stem contains names forbidden by Keras 3.
_DENSENET_STEM_NAMES = {
    "conv1/conv": "conv1_conv",
    "conv1/bn": "conv1_bn",
    "conv1/relu": "conv1_relu",
}

_AUTO_WEIGHTED_LAYER = re.compile(r"(conv2d|batch_normalization)(?:_(\d+))?")
_TF_SCOPE_SUFFIX = re.compile(r"(.+)_[1-9][0-9]*")


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
    """Translate legacy Functional graph configurations for Keras 3."""
    layers = config["layers"]
    nested_models = {
        layer["name"]
        for layer in layers
        if layer["class_name"] in {"Model", "Functional"}
    }
    for layer in layers:
        layer_name = layer["name"]
        layer_config = layer["config"]
        if layer["class_name"] == "SeparableConv2D":
            # Legacy SeparableConv2D also wrote the inherited Conv2D kernel
            # options. Its actual kernels use the depthwise/pointwise options.
            for field in ("kernel_initializer", "kernel_regularizer", "kernel_constraint"):
                layer_config.pop(field, None)
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


def _without_tf_scope_suffix(scope):
    match = _TF_SCOPE_SUFFIX.fullmatch(scope)
    if match is not None:
        original = match[1]
        return _DENSENET_STEM_NAMES.get(original, original)
    return scope


def _legacy_weights(group, *, legacy_tf_scopes):
    tensors = {}
    for name in _hdf_names(group, "layer_names"):
        layer = group.get(name)
        if not isinstance(layer, h5py.Group):
            raise ValueError(f"Missing legacy HDF5 layer {name!r}")
        layer_name = _DENSENET_STEM_NAMES.get(name, name)
        for weight in _hdf_names(layer, "weight_names"):
            weight_name = _weight_name(weight)
            scope, _, role = weight_name.rpartition("/")
            if legacy_tf_scopes and _without_tf_scope_suffix(scope) == layer_name:
                weight_name = f"{layer_name}/{role}"
            key = (layer_name, weight_name)
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


def _resolve_nested_tf_scopes(source, target):
    # The HDF5 group for a nested model holds tensors for all of its child
    # layers. Match a suffixed scope only to an exact child and role in that group.
    resolved = {}
    for key, dataset in source.items():
        if key not in target:
            layer, weight = key
            scope, _, role = weight.rpartition("/")
            candidate = (layer, f"{_without_tf_scope_suffix(scope)}/{role}")
            if candidate in target:
                key = candidate
        if key in resolved:
            raise ValueError(f"Duplicate legacy tensor identity {key}")
        resolved[key] = dataset
    return resolved


def _depthwise_kernel_aliases(model, target):
    # Keras 2 named DepthwiseConv2D's kernel depthwise_kernel. Other layers
    # such as SeparableConv2D still use that name in Keras 3.
    descendants = list(model.layers)
    depthwise_ids = set()
    while descendants:
        layer = descendants.pop()
        if isinstance(layer, keras.Model):
            descendants.extend(layer.layers)
        elif isinstance(layer, keras.layers.DepthwiseConv2D) and layer.built:
            depthwise_ids.add(id(layer.kernel))

    aliases = {}
    for (owner, path), variable in target.items():
        prefix, _, role = path.rpartition("/")
        if role == "kernel" and id(variable) in depthwise_ids:
            aliases[(owner, f"{prefix}/depthwise_kernel")] = (owner, path)
    return aliases


def assign_legacy_weights(model, group, *, existing_model=False):
    """Validate and read every tensor before changing any model weights."""
    writer = group.attrs.get("keras_version")
    if isinstance(writer, bytes):
        writer = writer.decode("utf-8")
    legacy_tf_scopes = isinstance(writer, str) and writer.endswith("-tf")

    source = _legacy_weights(group, legacy_tf_scopes=legacy_tf_scopes)
    target = _model_weights(model)
    if existing_model:
        source = _relative_layer_names(source)
        target = _relative_layer_names(target)
    aliases = {}
    if any(path.endswith("/depthwise_kernel") for _, path in source):
        aliases = _depthwise_kernel_aliases(model, target)
    if legacy_tf_scopes and source.keys() != target.keys():
        source = _resolve_nested_tf_scopes(source, target.keys() | aliases.keys())
    for legacy_key, modern_key in aliases.items():
        if legacy_key in source:
            if modern_key in source:
                raise ValueError(f"Duplicate legacy tensor identity {modern_key}")
            source[modern_key] = source.pop(legacy_key)
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

    if not existing_model:
        for variable, value in assignments:
            variable.assign(value)
        return

    # Reuse the staging list for undo data so peak storage grows by at most
    # one tensor. Include the current variable in case assign writes then fails.
    attempted = 0
    try:
        for i, (variable, value) in enumerate(assignments):
            assignments[i] = (variable, variable.numpy())
            attempted += 1
            variable.assign(value)
    except Exception as error:
        for i in reversed(range(attempted)):
            variable, original = assignments[i]
            try:
                variable.assign(original)
            except Exception as rollback_error:
                error.add_note(f"Could not restore {variable.path}: {rollback_error}")
        raise
