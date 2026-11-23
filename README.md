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
take ordered arrays of names or tables. Augmenters use imgaug classes and
DeepPoseKit's `FlipAxis`. Callbacks use Keras classes and DeepPoseKit's
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
sizes, and scheduler and stopping rules. The JSON also records the observed
epoch-end learning rates. A callback checkpoint reports the rate at the
point where that callback saves it. After a restart, observations may begin
at the resumed epoch. Republishing a completed backup created before this
feature embeds the verified TOML. Completed backups created with recipe
recording preserve both records when republished.

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

The locked environment uses `imgaug` 0.4.0. The `imaug` 0.4.3 fork
provides the same Python imports and ran the GSOR augmentation recipe on
NumPy 1.26 and 2.5. A shared seed produced different augmented images and
keypoints across the two distributions. The `imaug` release also installs
both GUI and headless OpenCV wheels, which share the `cv2` namespace. Keep
the locked dependency for existing recipes. See the
[imaug investigation](https://github.com/fhaefele/DeepPoseKit/issues/47)
and the [OpenCV wheel installation guidance](https://pypi.org/project/opencv-python/).

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
