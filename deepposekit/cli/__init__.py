"""DeepPoseKit command line and shell completion."""

import argparse
from functools import partial
from pathlib import Path
import sys

import argcomplete
from argcomplete.completers import FilesCompleter

from deepposekit import __version__
from deepposekit.cli.terminal import TerminalParser, color_mode


def _path(parser, name, *extensions, **options):
    action = parser.add_argument(name, type=Path, **options)
    if extensions:
        action.completer = FilesCompleter(allowednames=extensions)
    return action


def _count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def make_parser(color="auto"):
    parser_type = partial(TerminalParser, color=color)
    parser = parser_type(
        prog="deepposekit", description="Train and run pose estimation models."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", parser_class=parser_type
    )

    def command(name, group, description):
        parser.command_groups.setdefault(group, []).append((name, description))
        return commands.add_parser(name, description=description)

    def output_options(subparser):
        subparser.add_argument(
            "--overwrite", action="store_true", help="Replace an existing output."
        )

    def batch_option(subparser):
        subparser.add_argument(
            "--batch-size",
            type=_count,
            default=8,
            help="Frames per batch (default: 8).",
        )

    dataset = command("dataset", "Data and inspection", "Prepare annotation datasets.")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command", required=True, parser_class=parser_type
    )
    init = dataset_commands.add_parser(
        "init", help="Initialize an annotation file from video."
    )
    _path(init, "output", "h5", help="Output annotation HDF5 file.")
    _path(
        init, "--video", "avi", "mp4", "mov", "mkv", required=True, help="Input video."
    )
    _path(init, "--skeleton", "csv", required=True, help="Keypoint skeleton CSV.")
    init.add_argument(
        "--frame-step", type=_count, default=1, help="Sample every N frames."
    )
    init.add_argument("--gray", action="store_true", help="Read grayscale frames.")
    output_options(init)

    inspect = command(
        "inspect", "Data and inspection", "Show dataset or checkpoint metadata."
    )
    _path(
        inspect,
        "input",
        "h5",
        "hdf5",
        "keras",
        help="Annotation HDF5 file or checkpoint.",
    )
    inspect.add_argument(
        "--dataset", default="images", help="Image dataset key (default: images)."
    )

    train = command("train", "Model workflow", "Train or fine-tune from a TOML config.")
    _path(train, "config", "toml", help="Training configuration file.")
    _path(
        train,
        "--init-from",
        "h5",
        "hdf5",
        "keras",
        help="Full checkpoint to fine-tune.",
    )
    output_options(train)

    evaluate = command(
        "evaluate", "Model workflow", "Measure keypoint error on labeled frames."
    )
    _path(
        evaluate,
        "checkpoint",
        "h5",
        "hdf5",
        "keras",
        help="Full DeepPoseKit checkpoint.",
    )
    _path(evaluate, "annotations", "h5", help="Labeled holdout HDF5 file.")
    _path(evaluate, "--output", "json", help="Write metrics to JSON.")
    evaluate.add_argument(
        "--dataset", default="images", help="Image dataset key (default: images)."
    )
    batch_option(evaluate)
    output_options(evaluate)

    predict = command(
        "predict", "Model workflow", "Write keypoint predictions for a recording."
    )
    _path(
        predict,
        "checkpoint",
        "h5",
        "hdf5",
        "keras",
        help="Full DeepPoseKit checkpoint.",
    )
    _path(predict, "input", help="Input recording.")
    _path(predict, "output", "npy", help="Output NumPy array.")
    predict.add_argument(
        "--gray", action="store_true", help="Read grayscale using the built-in reader."
    )
    batch_option(predict)
    output_options(predict)
    _path(
        predict,
        "--custom-reader",
        "py",
        help="Python file with a create_reader factory.",
    )
    predict.add_argument(
        "--custom-reader-args",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help="Forward the remaining arguments to the custom reader.",
    )

    convert = command(
        "convert", "Model workflow", "Convert a legacy HDF5 checkpoint to .keras."
    )
    _path(convert, "input", "h5", "hdf5", help="Legacy full checkpoint or weights.")
    _path(convert, "output", "keras", help="Output Keras checkpoint.")
    _path(
        convert,
        "--config",
        "toml",
        help="Matching training config for legacy weights only.",
    )
    output_options(convert)

    command(
        "doctor",
        "Environment and tools",
        "Report package versions and available devices.",
    )
    completion = command(
        "completion", "Environment and tools", "Print shell completion code."
    )
    completion.add_argument(
        "shell", choices=("bash", "zsh"), help="Shell to generate code for."
    )
    return parser


def main(argv=None):
    """Parse and dispatch a command, delaying ML imports until execution."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = make_parser(color_mode(argv))
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "completion":
        sys.stdout.write(argcomplete.shellcode(["deepposekit"], shell=args.shell))
        return 0
    if args.command == "predict":
        if args.custom_reader_args is not None and args.custom_reader is None:
            parser.error("--custom-reader-args requires --custom-reader")
        if args.gray and args.custom_reader is not None:
            parser.error(
                "Pass grayscale settings through --custom-reader-args with a custom reader."
            )
    try:
        from deepposekit.cli import operations

        operations.run(args)
    except KeyboardInterrupt:
        parser.show_error("Interrupted.")
        return 130
    except Exception as error:
        parser.show_error(error)
        return 1
    return 0
