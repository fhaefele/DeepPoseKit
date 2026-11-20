"""Shared parsing of declarative component names and options."""

from pathlib import Path


def config_path(value, directory, option):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{option} must be a nonempty path")
    return directory / Path(value).expanduser()


def named_spec(value, what):
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return value["name"], {key: val for key, val in value.items() if key != "name"}
    raise ValueError(f"{what} must be a name or a table containing 'name'")


def named_list(values, what):
    if not isinstance(values, list):
        raise ValueError(f"{what} must be a list of named configurations")
    return (named_spec(value, what) for value in values)


def builtin_class(namespace, name, base):
    constructor = getattr(namespace, name, None)
    if (
        not isinstance(constructor, type)
        or not issubclass(constructor, base)
        or constructor is base
    ):
        raise ValueError(f"Unknown built-in {namespace.__name__} component: {name}")
    return constructor
