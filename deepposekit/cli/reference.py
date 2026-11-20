"""Render the effective CLI parser as versioned JSON or Markdown."""

import argparse
import json
from pathlib import Path
import shlex


def _children(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            descriptions = {
                choice.dest: choice.help for choice in action._get_subactions()
            }
            for name, child in action.choices.items():
                yield name, child, descriptions.get(name)
            return


def _usage(parser):
    formatter = argparse.HelpFormatter(parser.prog, width=100)
    formatter.add_usage(
        parser.usage, parser._actions, parser._mutually_exclusive_groups, prefix=""
    )
    return " ".join(formatter.format_help().split())


def _label(action):
    return action.option_strings[-1] if action.option_strings else action.dest


def _public(action):
    return (
        not isinstance(action, argparse._SubParsersAction)
        and action.help != argparse.SUPPRESS
    )


def _argument(action):
    nargs = action.nargs if action.nargs is not None else 1
    kind = type(action).__name__.lstrip("_").removesuffix("Action").lower()
    value_type = action.type
    type_name = getattr(value_type, "__name__", "string")
    bounds = {
        "_count": ("integer", {"minimum": 1}),
        "_index": ("integer", {"minimum": 0}),
        "_positive_real": ("number", {"exclusive_minimum": 0}),
    }
    display, constraint = bounds.get(
        type_name,
        (
            {"int": "integer", "float": "number", "str": "string"}.get(
                type_name, "string"
            ),
            {},
        ),
    )
    if value_type is Path:
        display = "path"
    if kind in {"storetrue", "storefalse"}:
        kind = "store_" + kind.removeprefix("store")
        display = "boolean"
    result = {
        "name": action.dest,
        "flags": action.option_strings,
        "action": kind,
        "nargs": "remainder" if nargs == argparse.REMAINDER else nargs,
        "required": action.required,
        "description": (action.help or "").replace("%(prog)s", "the command"),
    }
    if nargs != 0 or display == "boolean":
        result["type"] = display
    result.update(constraint)
    if action.metavar is not None:
        result["metavar"] = action.metavar
    if action.choices is not None:
        result["choices"] = list(action.choices)
    if action.default != argparse.SUPPRESS:
        result["default"] = (
            str(action.default) if isinstance(action.default, Path) else action.default
        )
    elif "--color" in action.option_strings:
        result["default"] = "auto"
    return result


def describe(parser, *, path=(), description=None, examples=()):
    actions = [action for action in parser._actions if _public(action)]
    children = list(_children(parser))
    descriptions = dict(
        (name, group)
        for group, entries in parser.command_groups.items()
        for name, _ in entries
    )
    node = {
        "path": list(path),
        "usage": _usage(parser),
        "description": parser.description or description or "",
        "arguments": [_argument(action) for action in actions],
        "mutually_exclusive": [
            {
                "required": group.required,
                "arguments": [
                    _label(action)
                    for action in group._group_actions
                    if action in actions
                ],
            }
            for group in parser._mutually_exclusive_groups
            if any(action in actions for action in group._group_actions)
        ],
        "notes": list(parser.reference_notes),
        "examples": [
            {"description": title, "argv": list(argv)}
            for title, argv in examples
            if path and tuple(argv[: len(path)]) == path and not children
        ],
        "commands": [],
    }
    for name, child, summary in children:
        item = describe(
            child, path=(*path, name), description=summary, examples=examples
        )
        if name in descriptions:
            item["group"] = descriptions[name]
        node["commands"].append(item)
    return node


def _nodes(node):
    yield node
    for child in node["commands"]:
        yield from _nodes(child)


def _escape_cell(value):
    return value.replace("|", r"\|").replace("\n", " ")


def markdown(document):
    command = document["command"]
    program = document["program"]
    nodes = list(_nodes(command))
    output = [
        f"# {program} CLI reference",
        "",
        f"Command version `{document['program_version']}`. Generated from the installed CLI parser.",
        "",
        "The checked examples use the inputs prepared by [the runnable CLI workflow](../examples/cli_workflow.py).",
        "",
        "## Commands",
        "",
    ]
    for node in nodes:
        if not node["path"]:
            continue
        label = " ".join(node["path"])
        anchor = "-".join((program, *node["path"]))
        output.append(f"- [`{label}`](#{anchor})")
    for node in nodes:
        command_name = " ".join((program, *node["path"]))
        output.extend(["", f"## {command_name}", ""])
        if node["description"]:
            output.extend([node["description"], ""])
        output.extend(["```text", node["usage"], "```", ""])
        if node["arguments"]:
            output.extend(
                [
                    "| Argument or flags | Required | Default | Description |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for option in node["arguments"]:
                names = option["flags"] or [option["name"]]
                label = ", ".join(f"`{name}`" for name in names)
                default = option.get("default")
                rendered_default = (
                    "—"
                    if default is None
                    else f"`{json.dumps(default, ensure_ascii=False)}`"
                )
                description = option["description"]
                if "choices" in option:
                    description += (
                        " Values "
                        + ", ".join(f"`{value}`" for value in option["choices"])
                        + "."
                    )
                output.append(
                    f"| {label} | {'Yes' if option['required'] else 'No'} | "
                    f"{_escape_cell(rendered_default)} | {_escape_cell(description)} |"
                )
            output.append("")
        for group in node["mutually_exclusive"]:
            selected = ", ".join(f"`{value}`" for value in group["arguments"])
            scope = "exactly one" if group["required"] else "at most one"
            output.append(f"Use {scope} of {selected}.")
        for note in node["notes"]:
            output.extend(["", note])
        for example in node["examples"]:
            output.extend(
                [
                    "",
                    example["description"],
                    "",
                    "```bash",
                    shlex.join([program, *example["argv"]]),
                    "```",
                ]
            )
    return "\n".join(output).rstrip() + "\n"


def render_help(parser, args, *, version, deepposekit_version, examples):
    selected = parser
    summary = None
    for token in args.path:
        children = {
            name: (child, description)
            for name, child, description in _children(selected)
        }
        if token not in children:
            parser.error(f"Unknown command {' '.join(args.path)!r}.")
        selected, summary = children[token]
    if args.format == "text":
        selected.print_help()
        return ""
    document = {
        "schema_version": 1,
        "program": parser.prog,
        "program_version": version,
        "deepposekit_version": deepposekit_version,
        "command": describe(
            selected, path=tuple(args.path), description=summary, examples=examples
        ),
    }
    if args.format == "json":
        return (
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return markdown(document)
