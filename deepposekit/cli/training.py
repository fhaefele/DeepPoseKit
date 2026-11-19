"""Declarative configuration for existing DeepPoseKit training APIs."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
from hashlib import sha256
from inspect import Parameter, signature
from math import isfinite
import json
import os
from pathlib import Path
import sys
import tomllib
import warnings

import h5py
import keras
import numpy as np

from deepposekit import callbacks
from deepposekit.cli.augmentations import build_augmenters
from deepposekit.cli.components import builtin_class, named_list, named_spec
from deepposekit.cli.operations import (
    annotation_summary,
    atomic_output,
    input_file,
    output_file,
)
from deepposekit.cli.terminal import TrainingProgress, colors_enabled
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import (
    DeepLabCut,
    LEAP,
    StackedDenseNet,
    StackedHourglass,
    load_model,
)
from deepposekit.models.loading import read_checkpoint_metadata
from deepposekit.utils.image import resolve_n_transitions

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
    for section, options, names in (
        (
            "model",
            model,
            ("subpixel", "pretrained", "upsampling", "batchnorm", "use_bias"),
        ),
        ("generator", generator, ("shuffle", "use_graph")),
    ):
        for key in names:
            if key in options and type(options[key]) is not bool:
                raise ValueError(f"[{section}].{key} must be a boolean")
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


def _data_and_generator(config, stored=None):
    annotation_summary(config.annotations, config.data_options.get("dataset", "images"))
    data = DataGenerator(str(config.annotations), **config.data_options)
    options = {}
    if stored is not None:
        allowed = _parameters(TrainingGenerator, "generator", "augmenter")
        options.update({key: stored[key] for key in allowed & stored.keys()})
    options.update(config.generator_options)
    if "augmenter" in options:
        options["augmenter"] = build_augmenters(data, options["augmenter"]) or None
    if "downsample_factor" in options:
        _integer(options["downsample_factor"], 0, "[generator].downsample_factor")
    sigma = options.get("sigma", 5)
    if type(sigma) not in {int, float} or not isfinite(sigma) or sigma <= 0:
        raise ValueError("[generator].sigma must be a finite positive number")
    return TrainingGenerator(data, **options)


def build_model(config, *, initialize_weights=True):
    """Build an uncompiled model with the geometry in a training config."""
    options = dict(config.model_options)
    if not initialize_weights:
        parameters = signature(MODELS[config.model_name]).parameters
        for key in {"pretrained_weights", "weights"} & parameters.keys():
            options[key] = None
    return MODELS[config.model_name](_data_and_generator(config), **options)


def _keras_option(value, namespace):
    if isinstance(value, list):
        return [_keras_option(item, namespace) for item in value]
    if not isinstance(value, dict):
        return value
    if not isinstance(value.get("name"), str):
        return {key: _keras_option(item, namespace) for key, item in value.items()}
    name, parameters = named_spec(value, namespace.__name__)
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


def _initial_weight_paths(config):
    return [
        Path(value)
        for name, value in config.model_options.items()
        if name in {"pretrained_weights", "weights"} and value not in {None, "imagenet"}
    ]


def _check_callback_path(path, reserved, *, directory=False):
    resolved = path.resolve()
    for other in reserved:
        if (
            resolved == other
            or directory
            and resolved in other.parents
            or other in resolved.parents
            or path.exists()
            and other.exists()
            and path.samefile(other)
        ):
            raise ValueError(
                f"Callback path conflicts with another input or output: {path}"
            )
    return resolved


class _TrainingCheckpoint(callbacks.ModelCheckpoint):
    """Apply CLI output protection to each expanded checkpoint destination."""

    def __init__(self, *, reserved, overwrite, **options):
        super().__init__(**options)
        self.reserved = reserved
        self.overwrite = overwrite
        self.saved = set()

    def _save_model(self, epoch, batch, logs):
        path = Path(self._get_file_path(epoch, batch, logs))
        if not self._should_save_model(epoch, batch, logs, str(path)):
            return
        resolved = _check_callback_path(path, self.reserved - self.saved)
        overwrite = self.overwrite or resolved in self.saved
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.save_weights_only:
                with atomic_output(path, overwrite) as temporary:
                    self.model.train_model.save_weights(temporary)
            else:
                self.model.save(path, overwrite=overwrite)
        except FileExistsError as error:
            raise FileExistsError(
                f"Checkpoint already exists: {path}. Use --overwrite or a new destination."
            ) from error
        self.reserved.add(resolved)
        self.saved.add(resolved)


class _TrainingBackup(keras.callbacks.BackupAndRestore):
    """Clean up the backup generation written by this training run."""

    def __init__(self, **options):
        super().__init__(**options)
        self._written = set()

    @property
    def _generations(self):
        return (
            (Path(self._weights_path), Path(self._prev_weights_path)),
            (
                Path(self._training_metadata_path),
                Path(self._prev_training_metadata_path),
            ),
        )

    @property
    def _completed_path(self):
        return Path(self.backup_dir) / "completed.weights.h5"

    @property
    def files(self):
        yield self._completed_path
        for generation in self._generations:
            yield from generation

    def store_completed(self, training_model, run_id):
        owned = self._written | {self._completed_path}
        self._completed_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output(self._completed_path, overwrite=True) as temporary:
            training_model.save_weights(temporary)
            with h5py.File(temporary, "r+") as weights:
                weights.attrs["deepposekit_training_run"] = run_id
                weights.attrs["deepposekit_backup_files"] = json.dumps(
                    sorted(path.name for path in owned)
                )
        self._written = owned

    def recover_completed(self, run_id):
        if run_id is None or not self._completed_path.is_file():
            return False
        with h5py.File(self._completed_path, "r") as weights:
            if weights.attrs.get("deepposekit_training_run") != run_id:
                return False
            owned = json.loads(weights.attrs.get("deepposekit_backup_files", "[]"))
        self._written = {path for path in self.files if path.name in owned}
        self._written.add(self._completed_path)
        return True

    def restore_completed(self, training_model):
        optimizer = training_model.optimizer
        if optimizer is not None and not optimizer.built:
            optimizer.build(training_model.trainable_variables)
        training_model.load_weights(self._completed_path)

    def _save_model(self):
        written = [current for current, _ in self._generations]
        if self.double_checkpoint:
            written.extend(
                previous for current, previous in self._generations if current.exists()
            )
        super()._save_model()
        self._written.update(written)

    def on_train_end(self, logs=None):
        # Keras finishes fitting before the CLI publishes the checkpoint.
        # Keep the recovery files until publication succeeds.
        return

    def finalize(self):
        if self.delete_checkpoint:
            for path in self._written:
                path.unlink(missing_ok=True)


def _discard_empty_directory(path):
    try:
        path.rmdir()
    except OSError:
        # Preserve any directory that still contains output or prior backup files.
        pass


def _discard_empty_file(path, created):
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if current.st_size == 0 and os.path.samestat(current, created):
        path.unlink(missing_ok=True)


def _reserve_callback_path(path, directory, resources):
    if directory:
        path.mkdir(parents=True, exist_ok=False)
        resources.callback(_discard_empty_directory, path)
    else:
        with path.open("xb") as file:
            created = os.fstat(file.fileno())
        resources.callback(_discard_empty_file, path, created)


def _callback_list(config, selected, overwrite, resources, *, run_id=None):
    protected = {
        path.resolve()
        for path in (
            config.output,
            config.source,
            config.annotations,
            *_initial_weight_paths(config),
        )
    }
    if selected is not None:
        protected.add(selected.resolve())
    reserved = set(protected)
    specs = []
    destinations = []
    for name, options in named_list(
        config.fit_options.get("callbacks", []), "[training].callbacks"
    ):
        namespace = (
            callbacks if name in {"Logger", "ModelCheckpoint"} else keras.callbacks
        )
        callback_type = builtin_class(namespace, name, keras.callbacks.Callback)
        if name == "TensorBoard":
            options.setdefault("log_dir", "logs")
        for key in ("filepath", "filename", "log_dir", "backup_dir"):
            if key not in options:
                continue
            path = _path(options[key], config.source.parent, f"{name}.{key}")
            options[key] = str(path)
            if name == "ModelCheckpoint":
                # Keras expands placeholders for each actual save.
                continue
            directory = key in {"log_dir", "backup_dir"}
            resolved = _check_callback_path(path, reserved, directory=directory)
            if not overwrite and os.path.lexists(path):
                raise FileExistsError(
                    f"{name}.{key} may replace an output: {path}. Use --overwrite or a new destination."
                )
            reserved.add(resolved)
            destinations.append((path, directory))
        specs.append((name, callback_type, options))

    if not overwrite:
        for path, directory in destinations:
            try:
                _reserve_callback_path(path, directory, resources)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Callback output is already in use: {path}"
                ) from error

    backups = {}
    for index, (name, _, options) in enumerate(specs):
        if name != "BackupAndRestore":
            continue
        callback = _TrainingBackup(**options)
        for path in callback.files:
            _check_callback_path(path, protected)
            if path.is_symlink() or path.exists() and not path.is_file():
                raise ValueError(f"Expected a regular backup file at {path}")
        backups[index] = callback

    # A publication retry must leave metrics and other training outputs intact.
    recovered = [
        backup for backup in backups.values() if backup.recover_completed(run_id)
    ]
    if recovered:
        return [], recovered

    result = []
    for index, (name, callback_type, options) in enumerate(specs):
        if index in backups:
            callback = backups[index]
        elif name == "ModelCheckpoint":
            callback = _TrainingCheckpoint(
                reserved=reserved, overwrite=overwrite, **options
            )
        else:
            callback = callback_type(**options)
        result.append(callback)
    return result, []


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
    architecture = pose["config"]
    model_type = MODELS[config.model_name]
    supplied = {
        name: value
        for name, value in config.model_options.items()
        if name not in {"pretrained_weights", "weights"}
    }
    if supplied:
        original = {
            name: architecture[name]
            for name in _parameters(model_type, "train_generator") & architecture.keys()
        }
        expected = model_type(None, skip_init=True, **original).get_config()
        candidate = original | supplied
        if "n_transitions" in candidate:
            candidate["n_transitions"] = resolve_n_transitions(
                candidate["n_transitions"], *architecture["image_shape"][:2]
            )
        actual = model_type(None, skip_init=True, **candidate).get_config()
        for key in sorted(supplied):
            if key not in actual or actual[key] != expected.get(key):
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


def _training_run_id(config, selected, output):
    digest = sha256(config.source.read_bytes())
    inputs = [config.source, config.annotations]
    if selected is not None:
        inputs.append(selected)
    elif config.model_name != "StackedDenseNet" or config.model_options.get(
        "pretrained"
    ):
        inputs.extend(_initial_weight_paths(config))
    identity = []
    for path in inputs:
        stat = path.stat()
        identity.append(
            (
                str(path.resolve()),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    digest.update(json.dumps([str(output.absolute()), identity]).encode())
    return digest.hexdigest()


def train(args):
    config = read_training_config(args.config)
    selected = input_file(args.init_from) if args.init_from is not None else None
    protected = [config.source, config.annotations, *_initial_weight_paths(config)]
    if selected is not None:
        protected.append(selected)
    output = output_file(config.output, ".keras", args.overwrite, *protected)
    colored = colors_enabled(sys.stderr, getattr(args, "color", "auto"))
    progress = TrainingProgress(sys.stderr, colored)
    with redirect_stdout(progress), ExitStack() as resources:
        resources.callback(progress.finish)
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
        recovered = []
        if "callbacks" in fit:
            run_id = _training_run_id(config, selected, output)
            fit["callbacks"], recovered = _callback_list(
                config, selected, args.overwrite, resources, run_id=run_id
            )
        backups = recovered or [
            callback
            for callback in fit.get("callbacks", ())
            if isinstance(callback, _TrainingBackup)
        ]
        if recovered:
            recovered[0].restore_completed(model.train_model)
        else:
            history = model.fit(**fit)
            loss = history.history.get("loss", [])
            if not len(loss) or not np.isfinite(loss).all():
                raise ValueError("Training did not produce a finite loss")
            for backup in backups:
                backup.store_completed(model.train_model, run_id)
        model.save(output, overwrite=args.overwrite)
        for backup in backups:
            backup.finalize()
    print(output)
