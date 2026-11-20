# deepposekit CLI reference

Command version `0.6.0`. Generated from the installed CLI parser.

The checked examples use the inputs prepared by [the runnable CLI workflow](../examples/cli_workflow.py).

## Commands

- [`dataset`](#deepposekit-dataset)
- [`dataset init`](#deepposekit-dataset-init)
- [`dataset append`](#deepposekit-dataset-append)
- [`dataset prelabel`](#deepposekit-dataset-prelabel)
- [`dataset merge`](#deepposekit-dataset-merge)
- [`dataset split`](#deepposekit-dataset-split)
- [`imgaug`](#deepposekit-imgaug)
- [`imgaug list`](#deepposekit-imgaug-list)
- [`imgaug toml`](#deepposekit-imgaug-toml)
- [`imgaug preview`](#deepposekit-imgaug-preview)
- [`predictions`](#deepposekit-predictions)
- [`predictions view`](#deepposekit-predictions-view)
- [`predictions trace`](#deepposekit-predictions-trace)
- [`predictions outliers`](#deepposekit-predictions-outliers)
- [`inspect`](#deepposekit-inspect)
- [`train`](#deepposekit-train)
- [`evaluate`](#deepposekit-evaluate)
- [`predict`](#deepposekit-predict)
- [`convert`](#deepposekit-convert)
- [`doctor`](#deepposekit-doctor)
- [`completion`](#deepposekit-completion)
- [`help`](#deepposekit-help)

## deepposekit

Train and run pose estimation models.

```text
deepposekit [-h] [--color {auto,always,never}] [--version] COMMAND ...
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `--version` | No | — | show program's version number and exit |


## deepposekit dataset

Prepare annotation datasets.

```text
deepposekit dataset [-h] [--color {auto,always,never}] {init,append,prelabel,merge,split} ...
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |


## deepposekit dataset init

Initialize an annotation file from video.

```text
deepposekit dataset init [-h] [--color {auto,always,never}] --video VIDEO --skeleton SKELETON [--frame-step FRAME_STEP] [--sample {stride,diverse}] [--count COUNT] [--seed SEED] [--gray] [--custom-reader CUSTOM_READER] [--custom-reader-args ...] [--batch-size BATCH_SIZE] [--overwrite] output
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `output` | Yes | — | Output annotation HDF5 file. |
| `--video` | Yes | — | Input video or recording read by a custom reader. |
| `--skeleton` | Yes | — | Keypoint skeleton CSV. |
| `--frame-step` | No | `1` | Sample every N frames. |
| `--sample` | No | `"stride"` | Frame selection method. Values `stride`, `diverse`. |
| `--count` | No | — | Number of frames to select with diverse sampling. |
| `--seed` | No | — | Seed for diverse sampling. |
| `--gray` | No | `false` | Read grayscale frames. |
| `--custom-reader` | No | — | Python file with a create_reader factory. |
| `--custom-reader-args` | No | — | Forward the remaining arguments to the custom reader. |
| `--batch-size` | No | `8` | Frames per batch (default: 8). |
| `--overwrite` | No | `false` | Replace an existing output. |


`--custom-reader-args` requires `--custom-reader` and forwards every remaining token. With a custom reader, pass grayscale settings through the reader arguments.

`--sample diverse` requires `--count`. Use `--seed` with diverse sampling.

The output is an initialized annotation HDF5 file with unreviewed keypoints.

Initialize an annotation file from grayscale video.

```bash
deepposekit dataset init unlabeled.h5 --video clip.avi --skeleton skeleton.csv --gray
```

## deepposekit dataset append

Append selected recording frames for annotation.

```text
deepposekit dataset append [-h] [--color {auto,always,never}] --source SOURCE --frames FRAMES --output OUTPUT [--dataset DATASET] [--gray] [--custom-reader CUSTOM_READER] [--custom-reader-args ...] [--batch-size BATCH_SIZE] [--overwrite] annotations
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `annotations` | Yes | — | Existing annotation HDF5 file. |
| `--source` | Yes | — | Source recording. |
| `--frames` | Yes | — | CSV with zero-based frame indices. |
| `--output` | Yes | — | Expanded annotation file. |
| `--dataset` | No | `"images"` | Image dataset key (default: images). |
| `--gray` | No | `false` | Read grayscale frames. |
| `--custom-reader` | No | — | Python file with a create_reader factory. |
| `--custom-reader-args` | No | — | Forward the remaining arguments to the custom reader. |
| `--batch-size` | No | `8` | Frames per batch (default: 8). |
| `--overwrite` | No | `false` | Replace an existing output. |


`--custom-reader-args` requires `--custom-reader` and forwards every remaining token. With a custom reader, pass grayscale settings through the reader arguments.

The CSV must have a `frame` column with zero-based indices. Appended rows start unreviewed.

Append the source frames listed in a CSV.

```bash
deepposekit dataset append labeled.h5 --source clip.avi --gray --frames frames.csv --output expanded.h5
```

## deepposekit dataset prelabel

Suggest labels for unreviewed keypoints.

```text
deepposekit dataset prelabel [-h] [--color {auto,always,never}] --model MODEL --output OUTPUT [--dataset DATASET] [--batch-size BATCH_SIZE] [--overwrite] annotations
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `annotations` | Yes | — | Annotation HDF5 file to prefill. |
| `--model` | Yes | — | Full DeepPoseKit checkpoint. |
| `--output` | Yes | — | Annotation file with suggestions. |
| `--dataset` | No | `"images"` | Image dataset key (default: images). |
| `--batch-size` | No | `8` | Frames per batch (default: 8). |
| `--overwrite` | No | `false` | Replace an existing output. |


Suggest keypoints for unlabeled frames.

```bash
deepposekit dataset prelabel unlabeled.h5 --model model.keras --output prelabeled.h5
```

## deepposekit dataset merge

Combine compatible annotation files in input order.

```text
deepposekit dataset merge [-h] [--color {auto,always,never}] --output OUTPUT [--dataset DATASET] [--overwrite] ANNOTATIONS [ANNOTATIONS ...]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `annotations` | Yes | — | Annotation files in the order to combine. |
| `--output` | Yes | — | Combined annotation HDF5 file. |
| `--dataset` | No | `"images"` | Image dataset key (default: images). |
| `--overwrite` | No | `false` | Replace an existing output. |


Pass at least two files. Image shapes, array dtypes, and skeletons must agree. Files are combined in input order.

Combine initialized annotation files in input order.

```bash
deepposekit dataset merge labeled.h5 unlabeled.h5 --output combined.h5
```

## deepposekit dataset split

Extract frames or partition an annotation file.

```text
deepposekit dataset split [-h] [--color {auto,always,never}] (--frames FRAMES | --at AT) (--output OUTPUT | --output-dir OUTPUT_DIR) [--dataset DATASET] [--overwrite] annotations
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `annotations` | Yes | — | An initialized annotation HDF5 file. |
| `--frames` | No | — | Zero-based indices and end-exclusive ranges, e.g. 0:100,150. |
| `--at` | No | — | Comma-separated indices of the first row in each new part. |
| `--output` | No | — | Annotation HDF5 file for --frames. |
| `--output-dir` | No | — | New directory for numbered --at parts. |
| `--dataset` | No | `"images"` | Image dataset key (default: images). |
| `--overwrite` | No | `false` | Replace an existing output. |

Use exactly one of `--frames`, `--at`.
Use exactly one of `--output`, `--output-dir`.

`--frames` requires `--output`. Frame indices are zero-based and ranges exclude the stop. The result preserves selection order, including repeats.

`--at` requires a new `--output-dir` and publishes consecutive parts together. Split points must be strictly increasing and inside the dataset. `--overwrite` applies only to a single output file.

Extract rows 0 through 2 and then row 5.

```bash
deepposekit dataset split combined.h5 --frames 0:3,5 --output subset.h5
```

Write consecutive parts beginning at rows 0, 4, and 8.

```bash
deepposekit dataset split combined.h5 --at 4,8 --output-dir parts
```

## deepposekit imgaug

Discover and preview training augmentations.

```text
deepposekit imgaug [-h] [--color {auto,always,never}] {list,toml,preview} ...
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |


## deepposekit imgaug list

List supported augmenter names.

```text
deepposekit imgaug list [-h] [--color {auto,always,never}]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |


List supported augmenters.

```bash
deepposekit imgaug list
```

## deepposekit imgaug toml

Print example TOML for an ordered pipeline.

```text
deepposekit imgaug toml [-h] [--color {auto,always,never}] AUGMENTER [AUGMENTER ...]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `names` | Yes | — | Augmenter names in application order. |


Print an ordered augmentation pipeline as TOML.

```bash
deepposekit imgaug toml Affine FlipAxis GaussianBlur
```

## deepposekit imgaug preview

Preview augmented images and keypoints.

```text
deepposekit imgaug preview [-h] [--color {auto,always,never}] [--config CONFIG] [--annotations ANNOTATIONS] [--augment TOML] [--frame FRAME] [--samples SAMPLES] [--seed SEED] [--rgb] [--output OUTPUT] [--overwrite]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `--config` | No | — | Training or minimal augmentation TOML. |
| `--annotations` | No | — | Annotation file from which to draw samples. |
| `--augment` | No | — | Inline TOML table. Repeat to compose transforms. |
| `--frame` | No | — | Zero-based index in the annotation dataset. |
| `--samples` | No | `6` | Number of augmented variants. |
| `--seed` | No | — | Seed for repeatable augmentations. |
| `--rgb` | No | `false` | Interpret stored color images as RGB. |
| `--output` | No | — | Save a PNG preview grid. |
| `--overwrite` | No | `false` | Replace an existing output. |


Save a seeded augmentation preview.

```bash
deepposekit imgaug preview --config training.toml --frame 0 --seed 17 --samples 2 --output preview.png
```

## deepposekit predictions

Review saved keypoints and find outliers.

```text
deepposekit predictions [-h] [--color {auto,always,never}] {view,trace,outliers} ...
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |


## deepposekit predictions view

Overlay predictions on the source recording.

```text
deepposekit predictions view [-h] [--color {auto,always,never}] [--skeleton SKELETON] [--start START] [--stop STOP] [--output OUTPUT] [--overwrite] --source SOURCE [--fps FPS] [--gray] [--custom-reader CUSTOM_READER] [--custom-reader-args ...] [--batch-size BATCH_SIZE] predictions
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `predictions` | Yes | — | Saved prediction array. |
| `--skeleton` | No | — | Optional skeleton CSV or annotation file. |
| `--start` | No | `0` | First frame (zero-based). |
| `--stop` | No | — | Stop before this frame. |
| `--output` | No | — | Save the review output to a file. |
| `--overwrite` | No | `false` | Replace an existing output. |
| `--source` | Yes | — | Source recording used for prediction. |
| `--fps` | No | `30.0` | Export playback rate (default: 30). |
| `--gray` | No | `false` | Read grayscale frames. |
| `--custom-reader` | No | — | Python file with a create_reader factory. |
| `--custom-reader-args` | No | — | Forward the remaining arguments to the custom reader. |
| `--batch-size` | No | `8` | Frames per batch (default: 8). |


`--custom-reader-args` requires `--custom-reader` and forwards every remaining token. With a custom reader, pass grayscale settings through the reader arguments.

Save an overlay on the input frames.

```bash
deepposekit predictions view predictions.npy --source clip.avi --gray --skeleton skeleton.csv --output overlay.avi
```

## deepposekit predictions trace

Plot x, y, confidence, and a keypoint trajectory.

```text
deepposekit predictions trace [-h] [--color {auto,always,never}] [--skeleton SKELETON] [--start START] [--stop STOP] [--output OUTPUT] [--overwrite] [--keypoint KEYPOINT | --keypoint-name KEYPOINT_NAME] predictions
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `predictions` | Yes | — | Saved prediction array. |
| `--skeleton` | No | — | Optional skeleton CSV or annotation file. |
| `--start` | No | `0` | First frame (zero-based). |
| `--stop` | No | — | Stop before this frame. |
| `--output` | No | — | Save the review output to a file. |
| `--overwrite` | No | `false` | Replace an existing output. |
| `--keypoint` | No | `"0"` | Keypoint index or name from a skeleton. |
| `--keypoint-name` | No | — | Select a skeleton name, including numeric names. |

Use at most one of `--keypoint`, `--keypoint-name`.

Plot a keypoint trace and its confidence.

```bash
deepposekit predictions trace predictions.npy --keypoint 0 --output trace.png
```

## deepposekit predictions outliers

Rank frames by confidence drops and coordinate jumps.

```text
deepposekit predictions outliers [-h] [--color {auto,always,never}] --output OUTPUT [--top TOP] [--min-drop MIN_DROP] [--min-jump MIN_JUMP] [--overwrite] predictions
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `predictions` | Yes | — | Saved prediction array. |
| `--output` | Yes | — | CSV of candidate frame indices and scores. |
| `--top` | No | `50` | Maximum candidates (default: 50). |
| `--min-drop` | No | `0.1` | Minimum confidence drop (default: 0.1). |
| `--min-jump` | No | `10.0` | Minimum distance in pixels (default: 10). |
| `--overwrite` | No | `false` | Replace an existing output. |


Rank suspicious frames and export their indices.

```bash
deepposekit predictions outliers predictions.npy --output candidates.csv
```

## deepposekit inspect

Show dataset or checkpoint metadata.

```text
deepposekit inspect [-h] [--color {auto,always,never}] [--dataset DATASET] input
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `input` | Yes | — | Annotation HDF5 file or checkpoint. |
| `--dataset` | No | `"images"` | Image dataset key (default: images). |


Inspect an annotation file.

```bash
deepposekit inspect labeled.h5
```

## deepposekit train

Train or fine-tune from a TOML config.

```text
deepposekit train [-h] [--color {auto,always,never}] [--init-from INIT_FROM] [--custom-generator CUSTOM_GENERATOR] [--custom-generator-args ...] [--overwrite] config
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `config` | Yes | — | Training configuration file. |
| `--init-from` | No | — | Full checkpoint to fine-tune. |
| `--custom-generator` | No | — | Python file with a create_generator factory. Overrides the TOML path. |
| `--custom-generator-args` | No | — | Forward the remaining arguments to the custom generator. Overrides the TOML arguments. |
| `--overwrite` | No | `false` | Replace an existing output. |


Configure a custom generator in `[data]` or with `--custom-generator`. `--custom-generator-args` replaces the TOML arguments and forwards every remaining token.

Train a model from a TOML file.

```bash
deepposekit train training.toml
```

## deepposekit evaluate

Measure keypoint error on labeled frames.

```text
deepposekit evaluate [-h] [--color {auto,always,never}] [--output OUTPUT] [--dataset DATASET] [--batch-size BATCH_SIZE] [--overwrite] checkpoint annotations
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `checkpoint` | Yes | — | Full DeepPoseKit checkpoint. |
| `annotations` | Yes | — | Labeled holdout HDF5 file. |
| `--output` | No | — | Write metrics to JSON. |
| `--dataset` | No | `"images"` | Image dataset key (default: images). |
| `--batch-size` | No | `8` | Frames per batch (default: 8). |
| `--overwrite` | No | `false` | Replace an existing output. |


Evaluate a checkpoint on labeled frames.

```bash
deepposekit evaluate model.keras holdout.h5 --output metrics.json
```

## deepposekit predict

Write keypoint predictions for recordings.

```text
deepposekit predict [-h] [--color {auto,always,never}] [--output-dir OUTPUT_DIR] [--recursive] [--batch-size BATCH_SIZE] [--overwrite] [--gray] [--custom-reader CUSTOM_READER] [--custom-reader-args ...] checkpoint PATH [PATH ...]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `checkpoint` | Yes | — | Full DeepPoseKit checkpoint. |
| `paths` | Yes | — | Input followed by an output .npy, or one or more inputs with --output-dir. |
| `--output-dir` | No | — | Write one .npy per recording and reader stream under this directory. |
| `--recursive` | No | `false` | Include subfolders of input directories. |
| `--batch-size` | No | `8` | Frames per batch (default: 8). |
| `--overwrite` | No | `false` | Replace an existing output. |
| `--gray` | No | `false` | Read grayscale frames. |
| `--custom-reader` | No | — | Python file with a create_reader factory. |
| `--custom-reader-args` | No | — | Forward the remaining arguments to the custom reader. |


`--custom-reader-args` requires `--custom-reader` and forwards every remaining token. With a custom reader, pass grayscale settings through the reader arguments.

Without `--output-dir`, pass one recording and one output `.npy`. Directory inputs and `--recursive` require `--output-dir`.

Predict on a grayscale video.

```bash
deepposekit predict model.keras clip.avi predictions.npy --gray --batch-size 2
```

## deepposekit convert

Convert a legacy HDF5 checkpoint to .keras.

```text
deepposekit convert [-h] [--color {auto,always,never}] [--config CONFIG] [--overwrite] input output
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `input` | Yes | — | Legacy full checkpoint or weights. |
| `output` | Yes | — | Output Keras checkpoint. |
| `--config` | No | — | Matching training config for legacy weights only. |
| `--overwrite` | No | `false` | Replace an existing output. |


## deepposekit doctor

Report package versions and available devices.

```text
deepposekit doctor [-h] [--color {auto,always,never}]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |


Check the installed versions and devices.

```bash
deepposekit doctor
```

## deepposekit completion

Print shell completion code.

```text
deepposekit completion [-h] [--color {auto,always,never}] {bash,zsh}
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `shell` | Yes | — | Shell to generate code for. Values `bash`, `zsh`. |


Generate Bash shell completion.

```bash
deepposekit completion bash
```

## deepposekit help

Show the installed command reference.

```text
deepposekit help [-h] [--color {auto,always,never}] [--format {text,markdown,json}] [COMMAND ...]
```

| Argument or flags | Required | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | No | — | show this help message and exit |
| `--color` | No | `"auto"` | Terminal colors (default: auto). Values `auto`, `always`, `never`. |
| `--format` | No | `"text"` | Reference output format (default: text). Values `text`, `markdown`, `json`. |
| `path` | Yes | — | Optional command and subcommand names. |


Print the installed dataset split contract as JSON.

```bash
deepposekit help --format json dataset split
```
