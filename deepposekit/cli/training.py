"""Declarative configuration for existing DeepPoseKit training APIs."""

from contextlib import redirect_stdout
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
import sys
import tomllib
import warnings

import imgaug.augmenters as iaa
import keras
import numpy as np

from deepposekit import FlipAxis, callbacks
from deepposekit.cli.operations import annotation_summary, input_file, output_file
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import (
    DeepLabCut,
    LEAP,
    StackedDenseNet,
    StackedHourglass,
    load_model,
)
from deepposekit.models.loading import read_checkpoint_metadata

MODELS = {
    model.__name__: model
    for model in (StackedDenseNet, StackedHourglass, LEAP, DeepLabCut)
}


@dataclass(frozen=True)
class TrainingConfig:
    source: Path
    annotations: Path
    data_options: dict
    generator_options: dict
    model_name: str
    model_options: dict
    compile_options: dict
    fit_options: dict
    output: Path | None


def _parameters(function, *excluded):
    return {
        name
        for name, value in signature(function).parameters.items()
        if name not in excluded
        and value.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }


def _validate_keys(values, allowed, section):
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {section} option(s): {', '.join(sorted(unknown))}")


def _section(source, name, allowed):
    data = source.get(name, {})
    if not isinstance(data, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    _validate_keys(data, allowed, f"[{name}]")
    return dict(data)


def _path(value, directory, option):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{option} must be a nonempty path")
    return directory / Path(value).expanduser()


def _integer(value, minimum, option):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{option} must be an integer >= {minimum}")


def read_training_config(path, *, for_weights=False):
    path = input_file(path).resolve()
    with path.open("rb") as file:
        source = tomllib.load(file)
    _validate_keys(
        source,
        {"data", "generator", "model", "compile", "training", "output"},
        "config",
    )
    data = _section(
        source, "data", _parameters(DataGenerator, "datapath") | {"annotations"}
    )
    annotations = _path(
        data.pop("annotations", None), path.parent, "[data].annotations"
    )
    generator = _section(
        source, "generator", _parameters(TrainingGenerator, "generator")
    )
    model = source.get("model", {})
    name = model.get("name", "StackedDenseNet") if isinstance(model, dict) else None
    if not isinstance(name, str) or name not in MODELS:
        raise ValueError(f"[model].name must be one of: {', '.join(MODELS)}")
    model = _section(
        source, "model", _parameters(MODELS[name], "train_generator") | {"name"}
    )
    model.pop("name", None)
    for key in ("pretrained_weights", "weights"):
        if key in model:
            value = model[key]
            if value == "none":
                model[key] = None
            elif value != "imagenet":
                model[key] = str(_path(value, path.parent, f"[model].{key}"))
    compile_options = _section(
        source, "compile", _parameters(keras.Model.compile, "self")
    )
    fit_keys = _parameters(MODELS[name].fit, "self") | {"max_queue_size"}
    fit_keys |= _parameters(
        keras.Model.fit,
        "self",
        "x",
        "y",
        "validation_data",
        "validation_split",
        "sample_weight",
    )
    fit_options = _section(source, "training", fit_keys)
    output_section = _section(source, "output", {"checkpoint"})
    output = None
    if not for_weights or "checkpoint" in output_section:
        output = _path(
            output_section.get("checkpoint"), path.parent, "[output].checkpoint"
        )
    if not for_weights or "batch_size" in fit_options:
        _integer(fit_options.get("batch_size"), 1, "[training].batch_size")
    for key in (
        "validation_batch_size",
        "epochs",
        "steps_per_epoch",
        "validation_steps",
        "max_queue_size",
    ):
        if key in fit_options:
            _integer(fit_options[key], 1, f"[training].{key}")
    for key in ("initial_epoch", "n_workers"):
        if key in fit_options:
            _integer(fit_options[key], 0, f"[training].{key}")
    if fit_options.get("initial_epoch", 0) >= fit_options.get("epochs", 1):
        raise ValueError("initial_epoch must be less than epochs")
    if (
        "use_multiprocessing" in fit_options
        and type(fit_options["use_multiprocessing"]) is not bool
    ):
        raise ValueError("[training].use_multiprocessing must be a boolean")
    return TrainingConfig(
        path,
        annotations,
        data,
        generator,
        name,
        model,
        compile_options,
        fit_options,
        output,
    )


def _named_spec(value, what):
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return value["name"], {key: val for key, val in value.items() if key != "name"}
    raise ValueError(f"{what} must be a name or a table containing 'name'")


def _named_list(values, what):
    if not isinstance(values, list):
        raise ValueError(f"{what} must be a list of named configurations")
    return (_named_spec(value, what) for value in values)


def _builtin_class(namespace, name, base):
    constructor = getattr(namespace, name, None)
    if (
        not isinstance(constructor, type)
        or not issubclass(constructor, base)
        or constructor is base
    ):
        raise ValueError(f"Unknown built-in {namespace.__name__} component: {name}")
    return constructor


def _augmenter(data, name, parameters):
    parameters = dict(parameters)
    for key in ("children", "then_list", "else_list"):
        if key in parameters:
            parameters[key] = [
                _augmenter(data, *spec)
                for spec in _named_list(parameters[key], "augmenter")
            ]
    if name == "FlipAxis":
        return FlipAxis(data, **parameters)
    return _builtin_class(iaa, name, iaa.Augmenter)(**parameters)


def _data_and_generator(config, stored=None):
    annotation_summary(config.annotations, config.data_options.get("dataset", "images"))
    data = DataGenerator(str(config.annotations), **config.data_options)
    options = {}
    if stored is not None:
        allowed = _parameters(TrainingGenerator, "generator", "augmenter")
        options.update({key: stored[key] for key in allowed & stored.keys()})
    options.update(config.generator_options)
    if "augmenter" in options:
        options["augmenter"] = [
            _augmenter(data, *spec)
            for spec in _named_list(options["augmenter"], "[generator].augmenter")
        ] or None
    if "downsample_factor" in options:
        _integer(options["downsample_factor"], 0, "[generator].downsample_factor")
    return TrainingGenerator(data, **options)


def build_model(config):
    """Build an uncompiled model with the geometry in a training config."""
    return MODELS[config.model_name](
        _data_and_generator(config), **config.model_options
    )


def _keras_option(value, namespace):
    if isinstance(value, list):
        return [_keras_option(item, namespace) for item in value]
    if not isinstance(value, dict):
        return value
    if not isinstance(value.get("name"), str):
        return {key: _keras_option(item, namespace) for key, item in value.items()}
    name, parameters = _named_spec(value, namespace.__name__)
    obj = namespace.get(name)
    if not hasattr(obj, "get_config"):
        if parameters:
            raise ValueError(
                f"{name} does not accept parameters. Use the built-in class name."
            )
        return obj
    config = obj.get_config()
    config.update(parameters)
    return type(obj).from_config(config)


def _callback_list(config, selected, overwrite):
    result = []
    protected = {
        config.output.resolve(),
        config.source.resolve(),
        config.annotations.resolve(),
    }
    if selected is not None:
        protected.add(selected.resolve())
    destinations = set()
    for name, options in _named_list(
        config.fit_options.get("callbacks", []), "[training].callbacks"
    ):
        namespace = (
            callbacks if name in {"Logger", "ModelCheckpoint"} else keras.callbacks
        )
        callback_type = _builtin_class(namespace, name, keras.callbacks.Callback)
        if name == "TensorBoard":
            options.setdefault("log_dir", "logs")
        for key in ("filepath", "filename", "log_dir", "backup_dir"):
            if key not in options:
                continue
            path = _path(options[key], config.source.parent, f"{name}.{key}")
            resolved = path.resolve()
            directory = key in {"log_dir", "backup_dir"}
            if any(
                resolved == other
                or directory
                and resolved in other.parents
                or other in resolved.parents
                for other in protected | destinations
            ):
                raise ValueError(
                    f"{name}.{key} conflicts with another input or output: {path}"
                )
            if not overwrite and (
                "{" in path.name or path.exists() or path.is_symlink()
            ):
                raise FileExistsError(
                    f"{name}.{key} may replace an output: {path}. Use --overwrite or a new destination."
                )
            destinations.add(resolved)
            options[key] = str(path)
        result.append(callback_type(**options))
    return result


def _available_steps(model, fit):
    generator = model.train_generator
    train_steps = generator.n_train // fit["batch_size"]
    validation_steps = generator.n_validation // fit.get("validation_batch_size", 1)
    if train_steps == 0 or fit.get("steps_per_epoch", train_steps) > train_steps:
        raise ValueError(
            "batch_size and steps_per_epoch exceed the number of available training samples"
        )
    if generator.n_validation and validation_steps == 0:
        raise ValueError(
            "validation_batch_size exceeds the number of validation samples"
        )
    if fit.get("validation_steps", validation_steps) > validation_steps:
        raise ValueError(
            "validation_steps exceeds the number of available validation batches"
        )


def _fine_tune(config, source):
    metadata = read_checkpoint_metadata(source)
    pose = metadata["pose_model_config"]
    if pose["class_name"] != config.model_name:
        raise ValueError(
            f"[model].name does not match checkpoint model {pose['class_name']}"
        )
    for key, value in config.model_options.items():
        if key not in pose["config"] or pose["config"][key] != value:
            raise ValueError(
                f"[model].{key} must match the checkpoint or be omitted when fine-tuning"
            )
    stored = metadata["train_generator_config"]["config"]
    if "skeleton" not in stored or "skeleton_names" not in stored:
        warnings.warn(
            "Checkpoint has incomplete skeleton metadata. Use annotations with the original keypoint order.",
            stacklevel=2,
        )
    generator = _data_and_generator(config, stored)
    return load_model(source, generator=generator, compile=False)


def train(args):
    config = read_training_config(args.config)
    selected = input_file(args.init_from) if args.init_from is not None else None
    protected = [config.source, config.annotations]
    if selected is not None:
        protected.append(selected)
    output = output_file(config.output, ".keras", args.overwrite, *protected)
    with redirect_stdout(sys.stderr):
        model = (
            build_model(config) if selected is None else _fine_tune(config, selected)
        )
        _available_steps(model, config.fit_options)
        compile_options = {"optimizer": "adam", "loss": "mse"} | config.compile_options
        for key, namespace in (
            ("optimizer", keras.optimizers),
            ("loss", keras.losses),
            ("metrics", keras.metrics),
            ("weighted_metrics", keras.metrics),
        ):
            if key in compile_options:
                compile_options[key] = _keras_option(compile_options[key], namespace)
        model.compile(**compile_options)
        fit = dict(config.fit_options)
        if "callbacks" in fit:
            fit["callbacks"] = _callback_list(config, selected, args.overwrite)
        history = model.fit(**fit)
        loss = history.history.get("loss", [])
        if not len(loss) or not np.isfinite(loss).all():
            raise ValueError("Training did not produce a finite loss")
        model.save(output, overwrite=args.overwrite)
    print(output)
