"""DeepPoseKit command line and shell completion."""

import argparse
from functools import partial
from pathlib import Path
import sys

import argcomplete
from argcomplete.completers import DirectoriesCompleter, FilesCompleter

from deepposekit import __version__
from deepposekit.cli.terminal import TerminalParser, color_mode

READER_ARGS = "--custom-reader-args"


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


def make_parser(color="auto", *, prog="deepposekit", configure_predict=None):
    parser_type = partial(TerminalParser, color=color)
    parser = parser_type(prog=prog, description="Train and run pose estimation models.")
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
        "predict", "Model workflow", "Write keypoint predictions for recordings."
    )
    _path(
        predict,
        "checkpoint",
        "h5",
        "hdf5",
        "keras",
        help="Full DeepPoseKit checkpoint.",
    )
    _path(
        predict,
        "paths",
        nargs="+",
        metavar="PATH",
        help="Input followed by an output .npy, or one or more inputs with --output-dir.",
    )
    _path(
        predict,
        "--output-dir",
        help="Write one .npy per recording and reader stream under this directory.",
    ).completer = DirectoriesCompleter()
    predict.add_argument(
        "--recursive",
        action="store_true",
        help="Include subfolders of input directories.",
    )
    predict.add_argument("--gray", action="store_true", help="Read grayscale frames.")
    batch_option(predict)
    output_options(predict)
    if configure_predict is None:
        _path(
            predict,
            "--custom-reader",
            "py",
            help="Python file with a create_reader factory.",
        )
        predict.add_argument(
            READER_ARGS,
            nargs=argparse.REMAINDER,
            metavar="ARG",
            help="Forward the remaining arguments to the custom reader.",
        )
    else:
        predict.set_defaults(
            custom_reader=None,
            custom_reader_args=None,
            prepare_prediction=configure_predict(predict),
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


def main(argv=None, *, prog="deepposekit", configure_predict=None):
    """Run the CLI, optionally extending predict for an embedding command.

    configure_predict replaces the generic reader flags. It receives the
    prediction parser and can return a callable that configures the reader
    from parsed arguments before standard prediction validation.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = make_parser(
        color_mode(argv), prog=prog, configure_predict=configure_predict
    )
    argcomplete.autocomplete(parser)
    # argparse treats "--" specially even inside a REMAINDER. Parse only the
    # DeepPoseKit prefix so a reader receives all remaining arguments unchanged.
    end = argv.index("--") if "--" in argv else len(argv)
    index = argv.index(READER_ARGS) if READER_ARGS in argv[:end] else len(argv)
    args, extra = parser.parse_known_args(argv[: index + 1])
    # A variadic positional stops at the first option. Keep accepting input
    # and output paths separated by flags, as in the original predict syntax.
    if args.command == "predict":
        stop = extra.index("--") if "--" in extra else len(extra)
        if all(not token.startswith("-") for token in extra[:stop]):
            args.paths.extend(map(Path, extra[:stop] + extra[stop + 1 :]))
            extra = []
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if index < len(argv):
        args.custom_reader_args = argv[index + 1 :]
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "completion":
        sys.stdout.write(argcomplete.shellcode([prog], shell=args.shell))
        return 0
    if args.command == "predict":
        if args.output_dir is None and (len(args.paths) != 2 or args.recursive):
            parser.error(
                "Specify an input and output .npy, or use --output-dir with input paths."
            )
        prepare_prediction = getattr(args, "prepare_prediction", None)
        if prepare_prediction is not None:
            prepare_prediction(args)
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
