"""DeepPoseKit command line and shell completion."""

import argparse
from functools import partial
from math import isfinite
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


def _index(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def _positive_real(value):
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def make_parser(
    color="auto",
    *,
    prog="deepposekit",
    version=__version__,
    configure_predict=None,
    configure_reader=None,
):
    parser_type = partial(TerminalParser, color=color)
    parser = parser_type(prog=prog, description="Train and run pose estimation models.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
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

    def reader_options(subparser, configure=None):
        subparser.add_argument(
            "--gray", action="store_true", help="Read grayscale frames."
        )
        if configure is None:
            _path(
                subparser,
                "--custom-reader",
                "py",
                help="Python file with a create_reader factory.",
            )
            subparser.add_argument(
                READER_ARGS,
                nargs=argparse.REMAINDER,
                metavar="ARG",
                help="Forward the remaining arguments to the custom reader.",
            )
        else:
            subparser.set_defaults(
                custom_reader=None,
                custom_reader_args=None,
                prepare_reader=configure(subparser),
            )

    def review_options(subparser, *extensions):
        _path(subparser, "predictions", "npy", help="Saved prediction array.")
        _path(
            subparser,
            "--skeleton",
            "csv",
            "h5",
            "hdf5",
            help="Optional skeleton CSV or annotation file.",
        )
        subparser.add_argument(
            "--start", type=_index, default=0, help="First frame (zero-based)."
        )
        subparser.add_argument("--stop", type=_count, help="Stop before this frame.")
        _path(
            subparser, "--output", *extensions, help="Save the review output to a file."
        )
        output_options(subparser)

    dataset = command("dataset", "Data and inspection", "Prepare annotation datasets.")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command", required=True, parser_class=parser_type
    )
    init = dataset_commands.add_parser(
        "init", help="Initialize an annotation file from video."
    )
    _path(init, "output", "h5", help="Output annotation HDF5 file.")
    _path(
        init,
        "--video",
        "avi",
        "mp4",
        "mov",
        "mkv",
        "h5",
        "hdf5",
        required=True,
        help="Input video or recording read by a custom reader.",
    )
    _path(init, "--skeleton", "csv", required=True, help="Keypoint skeleton CSV.")
    init.add_argument(
        "--frame-step", type=_count, default=1, help="Sample every N frames."
    )
    init.add_argument(
        "--sample",
        choices=("stride", "diverse"),
        default="stride",
        help="Frame selection method.",
    )
    init.add_argument(
        "--count", type=_count, help="Number of frames to select with diverse sampling."
    )
    init.add_argument(
        "--seed", type=_index, default=None, help="Seed for diverse sampling."
    )
    reader_options(
        init, partial(configure_reader, role="init") if configure_reader else None
    )
    batch_option(init)
    output_options(init)

    append = dataset_commands.add_parser(
        "append", help="Append selected recording frames for annotation."
    )
    _path(append, "annotations", "h5", "hdf5", help="Existing annotation HDF5 file.")
    _path(append, "--source", required=True, help="Source recording.")
    _path(
        append,
        "--frames",
        "csv",
        required=True,
        help="CSV with zero-based frame indices.",
    )
    _path(
        append,
        "--output",
        "h5",
        "hdf5",
        required=True,
        help="Expanded annotation file.",
    )
    append.add_argument(
        "--dataset", default="images", help="Image dataset key (default: images)."
    )
    reader_options(
        append, partial(configure_reader, role="append") if configure_reader else None
    )
    batch_option(append)
    output_options(append)

    prelabel = dataset_commands.add_parser(
        "prelabel", help="Suggest labels for unreviewed keypoints."
    )
    _path(
        prelabel, "annotations", "h5", "hdf5", help="Annotation HDF5 file to prefill."
    )
    _path(
        prelabel,
        "--model",
        "keras",
        "h5",
        "hdf5",
        required=True,
        help="Full DeepPoseKit checkpoint.",
    )
    _path(
        prelabel,
        "--output",
        "h5",
        "hdf5",
        required=True,
        help="Annotation file with suggestions.",
    )
    prelabel.add_argument(
        "--dataset", default="images", help="Image dataset key (default: images)."
    )
    batch_option(prelabel)
    output_options(prelabel)

    merge = dataset_commands.add_parser(
        "merge", help="Combine compatible annotation files in input order."
    )
    _path(
        merge,
        "annotations",
        "h5",
        "hdf5",
        nargs="+",
        metavar="ANNOTATIONS",
        help="Annotation files in the order to combine.",
    )
    _path(
        merge,
        "--output",
        "h5",
        "hdf5",
        required=True,
        help="Combined annotation HDF5 file.",
    )
    merge.add_argument(
        "--dataset", default="images", help="Image dataset key (default: images)."
    )
    output_options(merge)

    split = dataset_commands.add_parser(
        "split", help="Extract frames or partition an annotation file."
    )
    _path(
        split, "annotations", "h5", "hdf5", help="An initialized annotation HDF5 file."
    )
    selects = split.add_mutually_exclusive_group(required=True)
    selects.add_argument(
        "--frames", help="Zero-based indices and end-exclusive ranges, e.g. 0:100,150."
    )
    selects.add_argument(
        "--at", help="Comma-separated indices of the first row in each new part."
    )
    outputs = split.add_mutually_exclusive_group(required=True)
    _path(outputs, "--output", "h5", "hdf5", help="Annotation HDF5 file for --frames.")
    _path(
        outputs, "--output-dir", help="New directory for numbered --at parts."
    ).completer = DirectoriesCompleter()
    split.add_argument(
        "--dataset", default="images", help="Image dataset key (default: images)."
    )
    output_options(split)

    image_tools = command(
        "imgaug", "Data and inspection", "Discover and preview training augmentations."
    )
    aug = image_tools.add_subparsers(
        dest="imgaug_command", required=True, parser_class=parser_type
    )
    aug.add_parser("list", help="List supported augmenter names.")
    toml = aug.add_parser("toml", help="Print example TOML for an ordered pipeline.")
    toml.add_argument(
        "names",
        nargs="+",
        metavar="AUGMENTER",
        help="Augmenter names in application order.",
    )
    preview = aug.add_parser("preview", help="Preview augmented images and keypoints.")
    _path(preview, "--config", "toml", help="Training or minimal augmentation TOML.")
    _path(
        preview,
        "--annotations",
        "h5",
        "hdf5",
        help="Annotation file from which to draw samples.",
    )
    preview.add_argument(
        "--augment",
        action="append",
        metavar="TOML",
        help="Inline TOML table. Repeat to compose transforms.",
    )
    preview.add_argument(
        "--frame", type=_index, help="Zero-based index in the annotation dataset."
    )
    preview.add_argument(
        "--samples", type=_count, default=6, help="Number of augmented variants."
    )
    preview.add_argument(
        "--seed", type=_index, help="Seed for repeatable augmentations."
    )
    preview.add_argument(
        "--rgb", action="store_true", help="Interpret stored color images as RGB."
    )
    _path(preview, "--output", "png", help="Save a PNG preview grid.")
    output_options(preview)

    reviews = command(
        "predictions",
        "Data and inspection",
        "Review saved keypoints and find outliers.",
    )
    review = reviews.add_subparsers(
        dest="predictions_command", required=True, parser_class=parser_type
    )
    view = review.add_parser(
        "view", help="Overlay predictions on the source recording."
    )
    review_options(view, "avi", "mp4")
    _path(view, "--source", required=True, help="Source recording used for prediction.")
    view.add_argument(
        "--fps",
        type=_positive_real,
        default=30.0,
        help="Export playback rate (default: 30).",
    )
    reader_options(
        view, partial(configure_reader, role="view") if configure_reader else None
    )
    batch_option(view)
    trace = review.add_parser(
        "trace", help="Plot x, y, confidence, and a keypoint trajectory."
    )
    review_options(trace, "png")
    selector = trace.add_mutually_exclusive_group()
    selector.add_argument(
        "--keypoint", default="0", help="Keypoint index or name from a skeleton."
    )
    selector.add_argument(
        "--keypoint-name", help="Select a skeleton name, including numeric names."
    )
    outliers = review.add_parser(
        "outliers", help="Rank frames by confidence drops and coordinate jumps."
    )
    _path(outliers, "predictions", "npy", help="Saved prediction array.")
    _path(
        outliers,
        "--output",
        "csv",
        required=True,
        help="CSV of candidate frame indices and scores.",
    )
    outliers.add_argument(
        "--top", type=_count, default=50, help="Maximum candidates (default: 50)."
    )
    outliers.add_argument(
        "--min-drop",
        type=_positive_real,
        default=0.1,
        help="Minimum confidence drop (default: 0.1).",
    )
    outliers.add_argument(
        "--min-jump",
        type=_positive_real,
        default=10.0,
        help="Minimum distance in pixels (default: 10).",
    )
    output_options(outliers)

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
    batch_option(predict)
    output_options(predict)
    reader_options(predict, configure_predict)

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


def main(
    argv=None,
    *,
    prog="deepposekit",
    version=__version__,
    configure_predict=None,
    configure_reader=None,
    cuda_module=None,
):
    """Run the CLI, optionally extending predict for an embedding command.

    configure_predict replaces the generic reader flags. It receives the
    prediction parser and can return a callable that configures the reader
    from parsed arguments before standard prediction validation.
    configure_reader receives each single-source reader parser and its role.
    cuda_module names the installable module used to restart a CLI executable
    with its bundled CUDA paths. Explicit argv calls run in the caller's process.
    """
    executable = argv is None
    argv = sys.argv[1:] if executable else list(argv)
    parser = make_parser(
        color_mode(argv),
        prog=prog,
        version=version,
        configure_predict=configure_predict,
        configure_reader=configure_reader,
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
    prepare_reader = getattr(args, "prepare_reader", None)
    if prepare_reader is not None:
        prepare_reader(args)
    if hasattr(args, "custom_reader"):
        if args.custom_reader_args is not None and args.custom_reader is None:
            parser.error("--custom-reader-args requires --custom-reader")
        if args.gray and args.custom_reader is not None:
            parser.error(
                "Pass grayscale settings through --custom-reader-args with a custom reader."
            )
    if args.command == "dataset" and args.dataset_command == "init":
        if args.sample == "diverse" and args.count is None:
            parser.error("--sample diverse requires --count")
        if args.sample == "stride" and (
            args.count is not None or args.seed is not None
        ):
            parser.error("--count and --seed require --sample diverse")

    needs_cuda = args.command in {
        "predict",
        "evaluate",
        "train",
        "convert",
        "doctor",
    } or (args.command == "dataset" and args.dataset_command == "prelabel")
    try:
        if executable and cuda_module is not None and needs_cuda:
            from _deepposekit_gpu import prepare_cuda

            main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            if main_spec is not None and main_spec.name == f"{cuda_module}.__main__":
                launch = sys.orig_argv[1:]
            else:
                launch = ["-m", cuda_module, *argv]
            prepare_cuda(["-P", *launch])

        if args.command == "imgaug":
            from deepposekit.cli.augmentations import run

            run(args)
        else:
            from deepposekit.cli import operations

            if args.command == "doctor":
                args.command_name = prog
                args.command_version = version
            operations.run(args)
    except KeyboardInterrupt:
        parser.show_error("Interrupted.")
        return 130
    except Exception as error:
        parser.show_error(error)
        return 1
    return 0


def console_main():
    return main(cuda_module="deepposekit")
