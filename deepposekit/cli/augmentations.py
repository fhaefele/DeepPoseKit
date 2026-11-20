"""Discover and preview the same imgaug pipelines used by training."""

from inspect import signature
import json
import tomllib

import imgaug.augmenters as iaa
import numpy as np

from deepposekit.cli.components import builtin_class, named_list


_EXAMPLES = {
    "Affine": {"rotate": 5},
    "BlendAlpha": {
        "factor": 0.5,
        "foreground": [{"name": "GaussianBlur", "sigma": 0.5}],
    },
    "Sometimes": {"p": 0.5, "then_list": [{"name": "Affine", "rotate": 5}]},
    "FlipAxis": {"axis": 1, "p": 0.5},
    "GaussianBlur": {"sigma": 0.5},
    "Fliplr": {"p": 0.5},
    "Flipud": {"p": 0.5},
    "Resize": {"size": 1.0},
    "Add": {"value": 10},
    "Multiply": {"mul": 1.1},
}


def _augmenter(data, name, options):
    options = dict(options)
    for key in ("children", "then_list", "else_list", "foreground", "background"):
        if key in options:
            options[key] = [
                _augmenter(data, *spec)
                for spec in named_list(options[key], "augmenter")
            ]
    if name == "FlipAxis":
        from deepposekit.augment import FlipAxis

        return FlipAxis(data, **options)
    return builtin_class(iaa, name, iaa.Augmenter)(**options)


def build_augmenters(data, specifications):
    return [
        _augmenter(data, *spec)
        for spec in named_list(specifications, "[generator].augmenter")
    ]


def _list():
    names = {"FlipAxis"}
    for name, value in vars(iaa).items():
        if (
            not name.startswith("_")
            and isinstance(value, type)
            and issubclass(value, iaa.Augmenter)
            and value is not iaa.Augmenter
        ):
            names.add(name)
    print(*sorted(names), sep="\n")


def _example_options(name):
    options = _EXAMPLES.get(name, {})
    if name != "FlipAxis":
        constructor = builtin_class(iaa, name, iaa.Augmenter)
        try:
            signature(constructor).bind(**options)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} needs additional parameters. See the imgaug documentation for its options."
            ) from error
    # Signature defaults can still be invalid or rely on removed NumPy APIs.
    # Test the exact generated spec with images and keypoints before advertising it.
    points = np.array([[[8.0, 12.0], [20.0, 24.0]]])
    swap = np.array([1, 0])
    for channels in (1, 3):
        try:
            pipeline = iaa.Sequential([_augmenter(swap, name, options)])
            pipeline.seed_(17)
            pipeline(
                images=np.zeros((1, 32, 32, channels), dtype=np.uint8),
                keypoints=points.copy(),
            )
        except (
            AssertionError,
            AttributeError,
            ImportError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            last_error = error
        else:
            return options
    raise ValueError(
        f"No working starter TOML for {name} with the installed imgaug: {last_error}"
    ) from last_error


def _toml_value(value):
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items())
            + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(map(_toml_value, value)) + "]"
    return json.dumps(value, ensure_ascii=False)


def _toml(names):
    examples = [_toml_value({"name": name, **_example_options(name)}) for name in names]
    print("[generator]")
    print("augmenter = [")
    for example in examples:
        print(f"  {example},")
    print("]")


def _inputs(args):
    from deepposekit.cli.data_source import read_data_source

    data = {}
    generator = {}
    if args.config is not None:
        with args.config.open("rb") as file:
            config = tomllib.load(file)
        data = config.get("data", {})
        generator = config.get("generator", {})
        if not isinstance(generator, dict):
            raise ValueError("[generator] must be a TOML table")
    directory = args.config.resolve().parent if args.config is not None else None
    source = read_data_source(data, directory, annotations=args.annotations)
    specifications = (
        [tomllib.loads(f"augmenter = {entry}")["augmenter"] for entry in args.augment]
        if args.augment is not None
        else generator.get("augmenter")
    )
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("Provide --augment or [generator].augmenter in --config")
    if args.augment is not None and not all(
        isinstance(spec, dict) for spec in specifications
    ):
        raise ValueError("Each --augment must be a TOML inline table containing 'name'")
    return source, specifications


def _variants(pipeline, image, points, samples):
    yield "Original", image[0], points[0]
    for index in range(1, samples + 1):
        augmented, keypoints = pipeline(images=image.copy(), keypoints=points.copy())
        yield f"Sample {index}", augmented[0], keypoints[0]


def _draw_panel(axis, image, points, edges, title, *, rgb):
    height, width = image.shape[:2]
    if image.shape[-1] == 1:
        axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=255)
    else:
        if not rgb and image.shape[-1] in (3, 4):
            order = [2, 1, 0] if image.shape[-1] == 3 else [2, 1, 0, 3]
            image = image[..., order]
        axis.imshow(image)
    valid = np.all(
        np.isfinite(points) & (points >= 0) & (points < (width, height)), axis=1
    )
    for first, second in edges:
        if valid[first] and valid[second]:
            axis.plot(*points[[first, second]].T, color="#f0c054", linewidth=1.2)
    axis.scatter(*points[valid].T, c="#62ff84", s=14)
    axis.set_title(title)
    axis.set_axis_off()


def _preview(args):
    from deepposekit.cli.data_source import open_data
    from deepposekit.cli.operations import output_file
    from deepposekit.cli.render import require_display

    source, specifications = _inputs(args)
    output = None
    overwrite = args.overwrite
    if args.output is None:
        require_display()
    else:
        target = args.output
        if source.custom_generator is not None and source.path.is_dir():
            directory = source.path.resolve()
            # Check intermediate ancestors before resolving the final parent.
            # A dataset can reach files outside its root through a directory symlink.
            through_dataset = any(
                parent.resolve().is_relative_to(directory)
                for parent in target.absolute().parents
            )
            target = target.parent.resolve() / target.name
            if through_dataset or target.resolve().is_relative_to(directory):
                if target.exists() or target.is_symlink():
                    raise ValueError(
                        f"Preview output would replace a file in the custom data source: {target}"
                    )
                # Creation stays exclusive even if another process creates the file first.
                overwrite = False
        inputs = source.inputs if args.config is None else (*source.inputs, args.config)
        output = output_file(target, ".png", overwrite, *inputs)
    with open_data(source, full=True) as data:
        if source.custom_generator is None:
            mode = source.options.get("mode", "annotated")
            eligible = {
                "annotated": data.annotated_index,
                "unannotated": data.unannotated_index,
                "full": data.index,
            }[mode]
            if args.frame is None and mode == "full" and data.n_annotated:
                eligible = data.annotated_index
            selection = f"{mode} mode"
        else:
            eligible = range(len(data))
            selection = "the custom generator"

        if not len(eligible):
            raise ValueError(f"No frames are available in {selection}")
        frame = int(eligible[0]) if args.frame is None else args.frame
        if frame not in eligible:
            raise ValueError(f"Frame {frame} is unavailable in {selection}")
        _render_preview(args, data, frame, specifications, output, overwrite)


def _render_preview(args, data, frame, specifications, output, overwrite):
    from matplotlib import pyplot as plt

    from deepposekit.cli.render import display_or_save, skeleton_edges

    pipeline = iaa.Sequential(build_augmenters(data, specifications))
    if args.seed is not None:
        pipeline.seed_(args.seed)
    image = data.get_images([frame])
    points = data.get_keypoints([frame]).astype(float)
    points[np.any(points < 0, axis=-1)] = np.nan
    edges = skeleton_edges(data.graph)
    columns = min(4, args.samples + 1)
    rows = (args.samples + columns) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
        layout="constrained",
    )
    try:
        for axis, (title, panel, coordinates) in zip(
            axes.flat, _variants(pipeline, image, points, args.samples), strict=False
        ):
            _draw_panel(axis, panel, coordinates, edges, title, rgb=args.rgb)
        for axis in list(axes.flat)[args.samples + 1 :]:
            axis.set_axis_off()
        figure.suptitle(f"Frame {frame}")
        display_or_save(figure, output, overwrite)
    finally:
        plt.close(figure)


def run(args):
    match args.imgaug_command:
        case "list":
            _list()
        case "toml":
            _toml(args.names)
        case "preview":
            _preview(args)
