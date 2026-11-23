<p align="center">
<img src="assets/deepposekit_logo.png" alt="DeepPoseKit logo" height="240">
</p>

# DeepPoseKit

DeepPoseKit trains models to track user-defined 2D keypoints in images and video.
This fork updates the original project's runtime and adds a CLI for working
with models, videos, and annotation datasets.
[Original DeepPoseKit](https://github.com/jgraving/DeepPoseKit) · [DeepPoseKit paper](https://doi.org/10.7554/eLife.47994)

## Changes in this fork

- **Updated runtime.** Use Python 3.12, TensorFlow 2.21, and Keras 3 with
  locked uv installs and optional NVIDIA GPU support on Linux and WSL2.
  [Installation](#installation)
- **Legacy model migration.** Load supported legacy HDF5 checkpoints,
  convert them to `.keras`, or fine-tune them with a fresh optimizer.
  Import standalone legacy StackedDenseNet weights with a matching model configuration.
  [Checkpoint migration](#import-a-full-checkpoint-and-predict) · [StackedDenseNet weights](#import-legacy-stackeddensenet-weights)
- **Training and batch prediction from the CLI.** Train from TOML, evaluate
  saved models, and predict across videos or directories with built-in or
  custom readers. [Command line](#command-line)
- **Dataset preparation and prediction review.** Preview training augmentations,
  inspect video overlays and keypoint traces, sample and prelabel frames, and
  merge or split annotation datasets. Use an external annotator for interactive labeling.
  [Prediction review](#review-and-refine-predictions)

## Installation

On a minimal Ubuntu 24.04 or WSL2 install, run `sudo apt update && sudo apt install libgl1 libglib2.0-0t64` for OpenCV.

```bash
git clone https://github.com/fhaefele/DeepPoseKit.git
cd DeepPoseKit
uv sync --locked
uv run --locked python -c "from deepposekit.models import StackedDenseNet"
```

### NVIDIA GPU on Linux or WSL2

Install the `gpu` extra on x86-64 Linux or WSL2 to add the CUDA libraries and
compiler. `deepposekit` sets up those paths automatically. TensorFlow uses a
visible NVIDIA GPU when available and runs on CPU when it sees no GPU.

```bash
uv sync --locked --extra gpu
uv run --locked deepposekit doctor
```

`doctor` reports the command, DeepPoseKit, Python, TensorFlow, and Keras
versions, the installed CUDA runtime, and whether TensorFlow can see a GPU.
Set `CUDA_VISIBLE_DEVICES=-1` to run a command on CPU.
[TensorFlow GPU installation guide](https://www.tensorflow.org/install/pip)

## Command line

From a checkout, run `uv run --locked deepposekit --help` or
`uv run --locked python -m deepposekit --help`. To install the CLI or run it
without a checkout, replace `CLI_REF` with a fork tag or commit containing the CLI.

```bash
fork='git+https://github.com/fhaefele/DeepPoseKit.git@CLI_REF'
uv tool install --python 3.12 "$fork"
deepposekit --help
uvx --python 3.12 --from "$fork" deepposekit --help
```

For a GPU tool install, select the extra once and use the same commands.

```bash
uv tool install --python 3.12 "deepposekit[gpu] @ $fork"
deepposekit train training.toml
```

### Prepare, train, and evaluate

Use a skeleton CSV with `name`, `parent`, and `swap` columns. Leave a root's
`parent` and an unpaired keypoint's `swap` empty. Label the resulting HDF5 in
an external annotator.

```bash
deepposekit dataset init annotations.h5 --video video.avi --skeleton skeleton.csv --frame-step 20 --gray
deepposekit inspect annotations.h5
```

Save this as `training.toml` beside your labeled `annotations.h5`.

```toml
[data]
annotations = "annotations.h5"
dataset = "images"

[generator]
downsample_factor = 2
use_graph = true
sigma = 5
validation_split = 0.2
random_seed = 17
augmenter = [{ name = "Affine", rotate = -5 }]

[model]
name = "StackedDenseNet"
n_stacks = 1
growth_rate = 48
subpixel = true

[compile]
optimizer = { name = "Adam", learning_rate = 0.001 }
loss = "mse"
jit_compile = false

[training]
batch_size = 2
epochs = 1
n_workers = 1
callbacks = [{ name = "EarlyStopping", monitor = "val_loss", patience = 2 }]

[output]
checkpoint = "fly.keras"
```

```bash
deepposekit train training.toml
deepposekit inspect fly.keras
deepposekit evaluate fly.keras holdout.h5 --output metrics.json
```

All paths in the config are relative to the TOML file unless absolute. The
options use the Python argument names.

| Section | Options |
| --- | --- |
| `[data]` | `DataGenerator` options, including `dataset` and `mode`, with `annotations` supplying the HDF5 path. An optional `preprocessing` string records how the images were prepared or loaded. |
| `[generator]` | `TrainingGenerator` options such as `downsample_factor`, `use_graph`, `graph_scale`, `sigma`, `validation_split`, `shuffle`, and `random_seed`. |
| `[model]` | Constructor options for `StackedDenseNet` (the default), `StackedHourglass`, `LEAP`, or `DeepLabCut`. Set `pretrained_weights = "none"` or `weights = "none"` on models that expose those options to initialize without external weights. |
| `[compile]` | Keras `Model.compile` options. Omit the section for Adam and mean squared error. |
| `[training]` | DeepPoseKit `fit` options and Keras fit controls such as `initial_epoch` and `validation_freq`. Set `batch_size`. Omit `steps_per_epoch` to use all full training batches. Set `random_seed` to seed Keras and imgaug before model construction. The generator's own seed still governs its split when supplied. |
| `[output]` | Final `.keras` checkpoint path. |

Optimizer, loss, and metric specifications accept Keras names or tables
containing a built-in name and its parameters. `augmenter` and `callbacks`
take ordered arrays of names or tables. The default augmentation backend uses
imgaug classes and DeepPoseKit's `FlipAxis`. Callbacks use Keras classes and DeepPoseKit's
`Logger` and `ModelCheckpoint`. Use the Python API to supply custom objects.
Commands that create an output require `--overwrite` to replace it.
Checkpoint callback paths may use Keras placeholders such as `{epoch}`.

Use a dedicated directory for `BackupAndRestore`. If training finishes and
the final checkpoint cannot be written, retry the same command with
`--overwrite` to publish the saved training state. When cleanup is enabled,
the CLI removes the backup files after publication succeeds.

For fine-tuning, use compatible labeled annotations and a new output path in
`fine-tune.toml`. Set `[model].name` for other backbones. Omit architecture
and confidence map settings to inherit them from the checkpoint. The loaded
checkpoint supplies the model weights, and training uses a fresh optimizer. For older checkpoints with incomplete skeleton metadata,
check the original keypoint order and skeleton before fine-tuning.

```bash
deepposekit train fine-tune.toml --init-from best_model_densenet.h5
```

### Rerun training from a checkpoint recipe

`train` embeds the exact input TOML and a JSON run record in its final
`.keras` checkpoint and in full `.keras` checkpoints saved by
`ModelCheckpoint`. The annotation data stays in its source file.

1. Extract both records from the checkpoint you want to reproduce.

   ```bash
   deepposekit inspect fly.keras --training-config > recovered.toml
   deepposekit inspect fly.keras --training-settings > training-run.json
   ```

   `train` takes the recovered TOML as its positional argument. Use the JSON
   as a reference for the resolved inputs and any command-line overrides.
   Both extraction flags belong to `inspect`.

2. Check the inputs and choose new output paths. The JSON's `config_path`
   is the original TOML path. Put `recovered.toml` beside it or update its
   relative paths. Use `data.path` to locate the annotations or custom loader
   source, and keep the original dataset selection and preprocessing. A
   recorded `data.preprocessing` describes how the source images were
   prepared or loaded. Update `[output].checkpoint` and any checkpoint, log,
   or backup paths in `[training].callbacks` for the rerun.

3. Run training with the recovered TOML.

   ```bash
   deepposekit train recovered.toml
   ```

   If the original run fine-tuned a checkpoint, pass the path from
   `initialization.path` as `--init-from`. This is the original starting
   checkpoint. The `.keras` file you extracted from is the output of
   that run. For ImageNet or a weights file, use the initialization settings
   in `[model]` and update the weights path if it moved.

   If the original run overrode a custom generator from the command line,
   reapply `data.custom_generator` and the tokens in
   `data.custom_generator_args`. Run from the recorded `working_directory`
   if the loader arguments contain relative paths, or update those arguments
   for your current directory. Put `--custom-generator-args` last. For example,
   a rerun with both a starting checkpoint and loader overrides is

   ```bash
   deepposekit train recovered.toml --init-from original-start.keras \
     --custom-generator my_generator.py --custom-generator-args --split new-labels
   ```

The recovered TOML keeps the configured seeds, augmentation order, batch
sizes, and scheduler and stopping rules. The JSON records the augmentation
backend, its version when used, and the observed epoch-end learning rates.
A callback checkpoint reports the rate at the point where that callback saves it. After a restart, observations may begin
at the resumed epoch. Republishing a completed backup created before this
feature embeds the verified TOML. Completed backups created with recipe
recording preserve both records when republished.

### Experimental Albumentations recipe

DeepPoseKit uses imgaug by default. The prototype uses Albumentations 2.0.8,
the MIT-licensed release from the repository archived in July 2025.
AlbumentationsX is separately maintained and offers AGPL and commercial
licensing. Adopting it would require reviewing the terms for DeepPoseKit's
distribution and downstream users. [Albumentations archive](https://github.com/albumentations-team/albumentations) · [AlbumentationsX licensing](https://github.com/albumentations-team/AlbumentationsX/blob/main/LICENSING.md)

Albumentations and albucore declare a dependency on `opencv-python-headless`.
DeepPoseKit and imgaug use `opencv-python`. Both wheels provide `cv2` and
OpenCV's installation instructions call for using a single wheel. For the
experiment, reuse the locked OpenCV wheel in a disposable environment and
install the remaining dependencies explicitly. The default installation and
lockfile continue to use imgaug. [OpenCV wheel installation](https://pypi.org/project/opencv-python/#installation-and-usage)

```bash
export UV_PROJECT_ENVIRONMENT="$(mktemp -d)/.venv"
uv sync --locked
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    pydantic==2.14.4 simsimd==6.5.16 stringzilla==5.3.2
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
    albumentations==2.0.8 albucore==0.0.24
export NO_ALBUMENTATIONS_UPDATE=1
```

Set `[generator].augmentation_backend = "albumentations"` and use
Albumentations transform names and options. The following `[generator]`
recipe approximates the GSOR notebook pipeline. Combine it with your usual
data, model, training, and output sections.

```toml
[generator]
augmentation_backend = "albumentations"
downsample_factor = 2
use_graph = false
sigma = 5
validation_split = 0.1
random_seed = 1
augmenter = [
  { name = "FlipAxis", axis = 0, p = 0.5 },
  { name = "FlipAxis", axis = 1, p = 0.5 },
  { name = "Sequential", p = 0.5, transforms = [
    { name = "Affine", scale = { x = [0.95, 1.05], y = [0.95, 1.05] }, translate_percent = { x = [-0.05, 0.05], y = [-0.05, 0.05] }, shear = { x = [-8, 8], y = [-8, 8] }, interpolation = 1, border_mode = 0, fill = 0, p = 1.0 },
    { name = "Affine", scale = [0.8, 1.2], keep_ratio = true, interpolation = 1, border_mode = 0, fill = 0, p = 1.0 },
    { name = "Perspective", scale = [0.01, 0.05], keep_size = true, interpolation = 1, border_mode = 0, fill = 0, p = 1.0 },
    { name = "CLAHE", clip_limit = [1, 4], tile_grid_size = [5, 5], p = 1.0 },
    { name = "OneOf", p = 1.0, transforms = [
      { name = "GaussianBlur", sigma_limit = [0.0, 3.0], blur_limit = 0, p = 1.0 },
      { name = "MotionBlur", blur_limit = [3, 11], allow_shifted = false, p = 1.0 },
      { name = "Defocus", radius = [3, 6], alias_blur = [0.1, 0.5], p = 1.0 },
    ] },
  ] },
  { name = "Affine", rotate = [-30, 30], interpolation = 1, border_mode = 0, fill = 0, p = 1.0 },
]
```

| GSOR operation | Mapping and differences |
| --- | --- |
| Vertical then horizontal `FlipAxis` | Independent 50% draws, same order and pixel reflection formula. Each applied flip uses the dataset's swap indices. |
| `Sometimes(0.5, ...)` and `OneOf` | `Sequential(p=0.5)` runs all children in order when selected. `OneOf(p=1.0)` with equal child weights chooses each blur one-third of the time inside that group, one-sixth overall. |
| Two affines inside the group | Independent x/y scale 0.95–1.05 and translation ±5%, then isotropic scale 0.8–1.2. Legacy shear shares one ±8° draw across axes. The prototype draws each axis independently. Albumentations truncates fractional translations to integer pixels and computes its own affine matrix. |
| Perspective | Corner jitter scale 0.01–0.05 and resize back to the input size in both. The corner sampling and matrix and resize conventions differ. The prototype explicitly uses bilinear interpolation and constant zero fill. |
| CLAHE | Clip limit 1–4 in both. Legacy samples a square tile grid of 3–7 per image. The prototype fixes it at 5×5. The grid values count tiles in OpenCV. |
| Gaussian blur | Sigma 0–3 in both. Automatic kernel sizing and filtering differ. |
| Motion blur | Legacy draws a kernel size uniformly from 3–10, then rounds even sizes up. The prototype draws odd sizes uniformly from 3–11 and uses an unshifted kernel. Both draw angles from 0–360° and direction from −1 to 1, with different kernels. |
| Defocus blur | Legacy severity 1–3 draws one of the radius and alias-blur pairs (3, 0.1), (4, 0.5), and (6, 0.5). Albumentations samples radius 3–6 and alias blur 0.1–0.5 independently. |
| Final rotation and affine resampling | Always rotates by a draw from ±30°. For all three affines, legacy samples nearest, linear, or cubic interpolation and constant, edge, symmetric, reflect, or wrap padding. The constant fill is sampled from 0–255. The prototype fixes these at linear interpolation and zero fill. Albumentations reflective affine modes duplicate labels, so the prototype excludes them. |

The adapter treats any negative or nonfinite input coordinate as a missing
annotation. Finite positive coordinates beyond the right or bottom edge
are passed through. Missing points return to their skeleton slots as NaN
with zero confidence maps, and edges connected to missing points are skipped.
Finite points that move outside the frame during augmentation retain their
coordinates on every side. If an intermediate warp clips an image feature,
a later warp can move its coordinate back inside the frame while the feature
remains absent. Validation batches use the original images and points.

Set `[training].random_seed` to seed the augmentation independently of the
generator's split seed. At each epoch start, Keras supplies the requested or
restored epoch number for newly generated batches. Keras can reuse prefetched
batches across partial epochs (`steps_per_epoch` shorter than a full data
pass). Setup prefetch under a distribution strategy and partial-epoch
prefetch can affect the draws a training run consumes. The checkpoint embeds
the TOML and records `augmentation_backend` and `augmentation_version` in
`inspect --training-settings`. Rerunning the TOML uses the selected backend.
Use the version in the run record to audit the installed package. The
experimental backend checks for 2.0.8 before running a recipe. Replay starts
a fresh augmentation stream. To migrate a recipe, rewrite the transforms
and preview the new distribution before training. Reused seed numbers
produce different draws between libraries.

```bash
uv run --no-sync deepposekit imgaug preview --config training.toml --seed 1 --output preview.png
uv run --no-sync deepposekit train training.toml
uv run --no-sync python -m unittest discover -s tests -k albumentations -v
```

#### GSOR comparison

The CPU throughput test timed image and keypoint transforms after loading
96 labeled 1024×1024 grayscale frames into memory. It used one OpenCV thread
on an Intel Core i7-14700 and three seeds after eight warm-up frames per
seed. Median throughput was 40.24 frames/s for imgaug and 83.31 frames/s
for Albumentations (2.07×). The measurements compare complete recipes,
including their different interpolation, border, and blur choices.

A geometric alignment check placed a synthetic marker at each selected
annotation coordinate before applying the flips, affines, and perspective
with bilinear and zero-fill settings. For visible markers ending away from
the frame edges, the median and maximum distances from the transformed
keypoints to the marker centroids were 0.210 and 0.506 px (81 markers)
for imgaug and 0.026 and 1.513 px (82 markers) for Albumentations.
Intermediate clipping erased three imgaug markers and one Albumentations
marker whose final coordinates were inside the frame.

The short training comparison resized the GSOR frames to 256×256 and used
the same 696/77 train/validation split, seeds of 1, initial weights, Adam
and mean squared error, and two-stack StackedDenseNet with growth rate 16,
three transitions, and no pretraining. Each run took three epochs of
48 batches with four samples per batch. The last-epoch validation
heatmap loss was 222.15 for imgaug and 223.68 for Albumentations. Mean
and median Euclidean keypoint errors over 1,105 valid validation points
were 30.62 and 24.09 px for imgaug and 59.04 and 38.96 px for
Albumentations. The original imgaug pipeline needed a process-local
`np.bool = np.bool_` alias for `OneOf` and `imagecorruptions==1.1.2`
with `setuptools==80.9.0` for the defocus branch.

Keep imgaug as the default. The prototype makes the GSOR recipe and
skeleton handling available for follow-up comparisons. The archived
release requires a manual OpenCV setup, and the single short run had
higher validation keypoint error. Revisit adoption if a maintained
dependency can share a single OpenCV wheel and larger matched runs
support migration.

### Train from a custom labeled dataset

A custom generator lets `train` and `imgaug preview` read labeled data through
the same Python module. Save a module such as `my_generator.py` that defines
`create_generator(path, *, argv)` and returns a
`deepposekit.io.BaseGenerator` instance. For example, this factory selects
one of several annotation files in a directory.

```python
import argparse

from deepposekit.io import DataGenerator


def create_generator(path, *, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "new-labels"), default="train")
    options = parser.parse_args(argv)
    return DataGenerator(str(path / f"{options.split}.h5"))
```

For a different format, return your own `BaseGenerator` subclass that
supplies image batches, keypoint coordinates, and their shapes. Its instances
must support deep copying for training and validation. If you enable
multiprocessing with spawn or forkserver workers, put the generator class in
an importable module. Thread workers can use a class defined in the loader
script. Define `graph`,
`swap_index`, and `skeleton_names` when your dataset has that metadata.

Replace the `[data]` table in the training example with the following and
keep the other training options. This example expects
`labeled-data/train.h5` beside the config.

```toml
[data]
source = "labeled-data/"
custom_generator = "my_generator.py"
custom_generator_args = ["--split", "train"]
```

```bash
deepposekit train training.toml
deepposekit imgaug preview --config training.toml --output preview.png
deepposekit train fine-tune.toml --init-from fly.keras \
  --custom-generator my_generator.py --custom-generator-args --split new-labels
```

For the fine-tuning command, copy the training config to `fine-tune.toml`,
set a new output checkpoint, and put your additional labeled data in
`labeled-data/new-labels.h5`.

`source` can be a file or directory. TOML paths are relative to the config.
A `--custom-generator` path is relative to the working directory. The CLI
flags override the corresponding TOML values. Put `--custom-generator-args`
last so the module receives every remaining token. Pass it with no values
to clear the configured arguments. An existing `[data].annotations` path can
also serve as the source. Pass any loader-specific dataset or mode settings
through the custom generator's arguments.

For augmentation preview, `--frame` selects a zero-based index in the
custom generator's sample order. Preview files inside a custom source directory
are created exclusively. Save previews outside that directory to replace them.
Use Python modules you trust, since DeepPoseKit executes the selected file.

### Convert and predict

```bash
deepposekit convert best_model_densenet.h5 fly-converted.keras
deepposekit predict fly-converted.keras video.avi predictions.npy --gray --batch-size 8
deepposekit convert legacy.weights-only.h5 fly-from-weights.keras --config legacy-training.toml
```

For a weights-only conversion, the config must match the original model and
annotation geometry. The [weights-only Python example](#import-legacy-stackeddensenet-weights)
lists the architecture options for the public fly checkpoint. Prediction accepts
both full legacy checkpoints and modern `.keras` files. The output is a NumPy
array of `(frames, keypoints, 3)` with `(x, y, confidence)` in input frame order.

A custom reader can add preprocessing or read another recording format.
The runnable [videoreader.py example](examples/videoreader.py) uses
`VideoReader` and adds grayscale and rotation options.

```bash
uv run --locked deepposekit predict model.keras video.avi predictions.npy \
  --batch-size 8 --custom-reader examples/videoreader.py \
  --custom-reader-args --gray --rotate 90
```

Run this command from the checkout root. For a tool install, download the
example and pass its path to `deepposekit predict`. Choose preprocessing that
matches the checkpoint's input shape and channel count. Prediction coordinates
refer to the transformed frames. Pass `--help` after `--custom-reader-args` to
show the example reader's options.

A custom reader module defines `create_reader(path, *, batch_size, argv)`.
For single-stream readers, DeepPoseKit passes this factory a `Path`, the
requested batch size, and a list of every argument after
`--custom-reader-args`. Put DeepPoseKit flags before that marker. Return an
iterable of NumPy arrays shaped `(batch, height, width, channels)` in frame
order, including the final short batch. DeepPoseKit closes the iterator and
reader when they provide `close()` methods. To support a different file
format, keep the factory signature and adapt how the example opens the
recording and yields frames.

#### Predict on multiple recordings

```bash
deepposekit predict model.keras a.avi b.avi --output-dir predictions
deepposekit predict model.keras recordings/ --output-dir predictions
deepposekit predict model.keras recordings/ --recursive --output-dir predictions
```

The checkpoint loads once per run. The CLI creates the output directory and
writes a `.npy` for each recording and selected stream. Directory scans preserve
paths relative to the input directory. In the recursive example above,
`recordings/session/clip.avi` produces `predictions/session/clip.npy`.
Explicit file inputs use their basenames.
The built-in reader scans common video extensions.

Colliding output names fail before inference, including names that differ
only by case. Existing outputs require `--overwrite`. If a recording fails
during inference, completed outputs from the run remain available.

Custom reader flags apply to every input. A reader may declare
`INPUT_EXTENSIONS` to filter directory scans. The example reader uses
`(".avi", ".mkv", ".mov", ".mp4")`. When a custom reader omits that setting,
all regular input files are candidates. Choose a separate output directory
for readers whose scans include `.npy` files.
For recordings with several streams, a reader can also define
`prediction_streams(argv)`. When this hook is present, the CLI uses its
mapping of stream names to factories accepting `(path, *, batch_size)`.
Use `--output-dir` with multiple streams. For example, a stream named
`Camera1` writes `recording.Camera1.npy`.

A downstream package can provide its own command name and reader options
through `deepposekit.cli.main(prog=..., configure_predict=...)`.
The callback supplies reader options on the shared prediction parser and can
translate them into `custom_reader` and `custom_reader_args`. Help and completion
use the downstream command name and the same parser.

### Preview training augmentations

```bash
deepposekit imgaug list
deepposekit imgaug toml Affine FlipAxis GaussianBlur
deepposekit imgaug preview --config training.toml \
  --frame 42 --samples 8 --seed 17 --output preview.png
```

Copy the generated array into `[generator].augmenter`. Some imgaug classes
need extra parameters or use APIs unavailable in the installed NumPy.
`imgaug toml` reports when it cannot produce a working starter. The transforms run
in the order listed. Preview uses the training pipeline on a labeled image
and overlays the corresponding keypoints on the original and each variant.
A minimal config uses the same sections as a full training config.

```toml
[data]
annotations = "fly-annotations.h5"

[generator]
augmenter = [
  { name = "Affine", rotate = 5 },
  { name = "FlipAxis", axis = 1, p = 0.5 },
  { name = "GaussianBlur", sigma = 0.5 },
]
```

Use repeatable `--augment` flags to try an ordered pipeline from the command
line. These flags replace the entire config pipeline when supplied together
with `--config`. Use `--annotations` to override the annotation path, and
`--output` to save a grid on a machine without a display. Frame indices are
zero-based rows in the annotation HDF5 file. Use `--rgb` for color datasets
stored in RGB order instead of OpenCV's BGR order.

```bash
deepposekit imgaug preview --annotations fly-annotations.h5 \
  --augment '{ name = "Affine", rotate = 15 }' \
  --augment '{ name = "FlipAxis", axis = 1, p = 0.5 }' \
  --frame 42 --output preview.png
```

### Review and refine predictions

```bash
deepposekit predictions view predictions.npy --source video.avi --gray
deepposekit predictions view predictions.npy --source video.avi --gray \
  --skeleton skeleton.csv --start 40 --stop 100 --output review.mp4 --fps 30
deepposekit predictions trace predictions.npy --keypoint 0 --output trace.png
```

The viewer overlays saved predictions on source frames. Use the same reader
and preprocessing from prediction, including `--custom-reader` and any forwarded
arguments. Use `--start` and `--stop` to select a zero-based, half-open frame
range. For headless review, export an AVI or MP4. `--fps` sets export playback
speed. The trace plots x, y, confidence, and the x–y trajectory from the
prediction array. A skeleton CSV or annotation HDF5 supplied with
`--skeleton` provides edges for overlays or keypoint names for traces. Use
`--keypoint-name` to select a numeric name that also matches a column index.

```bash
deepposekit predictions outliers predictions.npy --top 50 --output candidates.csv
deepposekit dataset append annotations.h5 --source video.avi --gray \
  --frames candidates.csv --output annotations-expanded.h5
deepposekit dataset prelabel annotations-expanded.h5 --model fly.keras \
  --output annotations-prelabeled.h5
```

The outlier CSV ranks frames by confidence drops and keypoint jumps. Edit
its `frame` column or remove rows to choose what to append. Indices are
zero-based and selected frames are appended in CSV order. Use the reader
and spatial settings used during prediction. Appended keypoints start
unreviewed. Prelabeling fills unreviewed keypoints with model suggestions
and preserves reviewed keypoints and review flags. For annotation files with
relative virtual datasets, save the new file beside the source file so its
backing paths still resolve.

```bash
deepposekit dataset init diverse.h5 --video video.avi --skeleton skeleton.csv \
  --gray --sample diverse --count 100 --frame-step 10 --seed 17
```

Diversity sampling clusters downsampled images and selects actual source
frames from across clusters. `--frame-step` sets the candidate stride.
`--count` sets the number of frames saved, and `--seed` makes the selection
repeatable.

### Combine or split annotation datasets

```bash
deepposekit dataset merge session-a.h5 session-b.h5 session-c.h5 --output combined.h5
deepposekit dataset split combined.h5 --frames '0:100,150,200:250' --output subset.h5
deepposekit dataset split combined.h5 --at 100,250 --output-dir parts/
```

Merge combines files in the order listed. The image shapes, array dtypes,
and skeletons must agree, including keypoint names and order when present.
Split uses zero-based rows in the annotation file. Ranges include the start
and stop before the end. Row selections preserve the requested order, including
repeated indices. Each index in `--at` starts a new part. The command
publishes all numbered files such as `part-001.h5` in a new directory on a
filesystem that supports exclusive directory renaming.

Both commands preserve coordinates, review flags, and skeleton metadata.
They write standalone DeepPoseKit annotation files. A merge retains ordinary
attributes shared by all sources and a split retains ordinary attributes from
its input. Use `--dataset` if the image key differs from `images`.
Use `--overwrite` to replace a single output file.

### Help, completion, and environment

The [CLI reference](docs/cli.md) lists every command and its flags. Use
the installed command to look up the reference for your version or return
a JSON object for automation.

```bash
deepposekit help --format json dataset split
deepposekit help --format markdown
```

JSON includes the executable and package versions, command and subcommand
paths, arguments, defaults, mutual exclusion groups, and example arguments.
Use `program` followed by the selected example's `argv` array to run it.

Run `deepposekit train --help` for command-specific flags and
`deepposekit doctor` for installed versions and detected devices. Help uses
terminal width and supports `--color auto|always|never`. Machine-readable
output is plain text.

In Bash, add the following to your shell configuration.

```bash
source <(deepposekit completion bash)
```

In Zsh, run the following after any existing `compinit` setup.

```zsh
autoload -Uz compinit
compinit
source <(deepposekit completion zsh)
```

### Run the checked CLI examples

From the checkout root, the workflow creates a tiny video and labeled
annotations, trains a model, and runs prediction, review, and dataset
commands. CI runs the same examples with the GPU hidden.

```bash
uv sync --locked
CUDA_VISIBLE_DEVICES=-1 uv run --locked python examples/cli_workflow.py
```

On Linux or WSL2 with an NVIDIA GPU, install the CUDA extra once and run
the workflow with a GPU check.

```bash
uv sync --locked --extra gpu
uv run --locked deepposekit doctor
uv run --locked python examples/cli_workflow.py --require-gpu
```

Use `--output-dir cli-demo` to keep the generated files in a new directory.

## Train from HDF5 annotations

Use `deepposekit.io.initialize_dataset` to prepare images and a keypoint
skeleton, then label the annotation HDF5 with an external annotation tool.
The package provides headless HDF5 readers, frame sampling, and keypoint
augmentation.
Leave the `parent` cell empty for each root keypoint in the skeleton CSV.

```python
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import StackedDenseNet

data = DataGenerator("annotations.h5")
training = TrainingGenerator(data, downsample_factor=2)
model = StackedDenseNet(training)
model.compile(optimizer="adam", loss="mse")
model.fit(batch_size=2, epochs=1)
model.save("model.keras")
```

| Model | Training generator downsample factor |
| --- | --- |
| `StackedDenseNet` | 2 or higher, with more transitions than downsampling steps |
| `StackedHourglass` | More model transitions than downsampling steps |
| `LEAP` | 0, with image height and width divisible by 4 |
| `DeepLabCut` | 2, 3, or 4 |

`DeepLabCut` is the original DeepPoseKit implementation. Its encoders include
ResNet, DenseNet, MobileNetV2, and Xception. Use `weights=None` to initialize
its encoder locally.

## Import a full checkpoint and predict

The example uses `best_model_densenet.h5` and `video.avi` from the public fly
dataset. Run the Python example with `uv run --locked python your_script.py`.
[Public example files](https://github.com/jgraving/DeepPoseKit-Data/tree/0aa5e3f5e8f9df63c48ba2bf491354472daa3e7e/datasets/fly)

```python
from deepposekit.io import VideoReader
from deepposekit.models import load_model

model = load_model("best_model_densenet.h5", compile=False)
video = VideoReader("video.avi", gray=True)
try:
    frames = video.read_batch(8, asarray=True)
finally:
    video.close()

predictions = model.predict(frames, batch_size=1)
model.save("fly.keras")
restored = load_model("fly.keras", compile=False)
```

A full DeepPoseKit `.h5` or `.hdf5` checkpoint stores model geometry and decoder
settings. The CPU suite compares imports and `.keras` reloads against pinned
legacy references for all four models.

For fine-tuning, pass a compatible annotation `DataGenerator` as
`generator=` when loading the model, then compile it with a fresh optimizer.
Match the checkpoint’s keypoint order, skeleton, image shape, and preprocessing.

## Import legacy StackedDenseNet weights

For a legacy Keras `save_weights` HDF5 export, construct the matching
`StackedDenseNet` first. The following options match the public fly model.
`pretrained_weights=None` builds its ImageNet-based encoder locally before
the imported weights are assigned.

```python
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import StackedDenseNet, load_legacy_weights

annotations = DataGenerator("annotation_data_release.h5")
training_data = TrainingGenerator(annotations, downsample_factor=3, use_graph=True, sigma=5)
model = StackedDenseNet(
    training_data,
    n_stacks=2,
    n_transitions=6,
    growth_rate=48,
    bottleneck_factor=1,
    compression_factor=0.5,
    pretrained=True,
    pretrained_weights=None,
    subpixel=True,
)
load_legacy_weights(model, "legacy.weights-only.h5")
model.save("fly-from-weights.keras")
```

The fixture runner recreates the weights-only example and records the legacy
outputs for comparison. [Legacy reference runner](tests/fixtures/capture_legacy_reference.py)

## Development and issues

Open an issue with your model class, checkpoint format, environment, and a
minimal example. [Fork issue tracker](https://github.com/fhaefele/DeepPoseKit/issues)

DeepPoseKit was developed by Jake Graving and Daniel Chae. [Upstream repository](https://github.com/jgraving/DeepPoseKit)

### CLI reference and checked examples

After changing the parser or its example arguments, regenerate the reference
from the checkout root.

```bash
uv run --locked deepposekit help --format markdown > docs/cli.md
```

The CPU suite checks the generated file against the parser. The separate CLI
example step executes the argument vectors shared by the reference and
`examples/cli_workflow.py`. Add new checked examples to that catalog and
provide their inputs in the workflow.

## License

Apache 2.0. [License](LICENSE)

# References

If you use DeepPoseKit for your research please cite [our open-access paper](http://paper.deepposekit.org):

    @article{graving2019deepposekit,
             title={DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning},
             author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
             journal={eLife},
             volume={8},
             pages={e47994},
             year={2019},
             publisher={eLife Sciences Publications Limited}
             url={https://doi.org/10.7554/eLife.47994},
             }

You can also read [our open-access preprint](http://preprint.deepposekit.org).

If you use the [imgaug package](https://github.com/aleju/imgaug) for data augmentation, please also consider [citing it](https://github.com/aleju/imgaug/blob/master/README.md#citation).

If you [use data](https://github.com/jgraving/DeepPoseKit#i-already-have-annotated-data) that was annotated with the DeepLabCut package (http://deeplabcut.org) for your research, be sure to [cite it](https://github.com/AlexEMG/DeepLabCut/blob/master/README.md#references).

Please also consider citing the relevant references for the pose estimation model(s) used in your research, which can be found in the documentation (i.e., [`StackedDenseNet`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedDenseNet.html#references), [`StackedHourglass`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedHourglass.html#references), [`DeepLabCut`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/DeepLabCut.html#references), [`LEAP`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/LEAP.html#references)).
