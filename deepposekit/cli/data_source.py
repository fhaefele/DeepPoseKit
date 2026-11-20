"""Load labeled samples for training and augmentation preview."""

from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_start_method
from pathlib import Path

from deepposekit.cli.components import config_path
from deepposekit.cli.operations import annotation_summary, custom_module
from deepposekit.io import BaseGenerator, DataGenerator


@dataclass(frozen=True)
class DataSource:
    path: Path
    options: dict
    custom_generator: Path | None
    argv: tuple[str, ...]

    @property
    def inputs(self):
        if self.custom_generator is None:
            return (self.path,)
        return self.path, self.custom_generator


def read_data_source(
    table,
    directory,
    *,
    annotations=None,
    custom_generator=None,
    custom_generator_args=None,
):
    if not isinstance(table, dict):
        raise ValueError("[data] must be a TOML table")
    options = dict(table)
    source = options.pop("source", None)
    annotation_path = options.pop("annotations", None)
    module = options.pop("custom_generator", None)
    argv = options.pop("custom_generator_args", None)
    unknown = set(options) - {"dataset", "mode"}
    if unknown:
        raise ValueError(f"Unknown [data] option(s): {', '.join(sorted(unknown))}")

    if custom_generator is None and module is not None:
        custom_generator = config_path(module, directory, "[data].custom_generator")
    if custom_generator_args is not None:
        argv = custom_generator_args
    if argv is not None and (
        not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv)
    ):
        raise ValueError("[data].custom_generator_args must be an array of strings")

    if custom_generator is not None:
        if options:
            raise ValueError(
                "[data].dataset and mode apply to the built-in HDF5 loader"
            )
        if annotations is not None:
            raise ValueError("Set [data].source to select input for a custom generator")
        if source is not None and annotation_path is not None:
            raise ValueError("Set only one of [data].source and [data].annotations")
        if source is None:
            path = config_path(
                annotation_path, directory, "[data].annotations or [data].source"
            )
        else:
            path = config_path(source, directory, "[data].source")
        return DataSource(path, options, custom_generator, tuple(argv or ()))

    if source is not None or argv is not None:
        raise ValueError(
            "[data].source and custom generator arguments require a custom generator"
        )
    if annotations is None:
        annotations = config_path(annotation_path, directory, "[data].annotations")
    dataset = options.get("dataset", "images")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("[data].dataset must be a nonempty string")
    if options.get("mode", "annotated") not in ("full", "annotated", "unannotated"):
        raise ValueError("[data].mode must be full, annotated, or unannotated")
    return DataSource(annotations, options, None, ())


@contextmanager
def open_data(source, *, full=False, process_workers=False):
    if source.custom_generator is None:
        options = dict(source.options)
        annotation_summary(source.path, options.get("dataset", "images"))
        if full:
            options["mode"] = "full"
        yield DataGenerator(str(source.path), **options)
        return

    if not source.path.exists():
        raise FileNotFoundError(f"Custom generator input not found: {source.path}")
    with custom_module(source.custom_generator, "generator") as module:
        factory = getattr(module, "create_generator", None)
        if not callable(factory):
            raise ValueError(
                "Custom generator must define create_generator(path, *, argv)"
            )
        data = factory(source.path, argv=list(source.argv))
        if not isinstance(data, BaseGenerator):
            raise TypeError(
                "create_generator must return a deepposekit.io.BaseGenerator instance"
            )
        try:
            if (
                process_workers
                and type(data).__module__ == module.__name__
                and get_start_method() != "fork"
            ):
                raise ValueError(
                    "Spawn and forkserver workers need an importable generator class. "
                    "Define the class in an installed module or set use_multiprocessing=false."
                )
            if not len(data):
                raise ValueError("The custom generator has no samples")
            yield data
        finally:
            close = getattr(data, "close", None)
            if callable(close):
                close()
