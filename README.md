<p align="center">
<img src="assets/deepposekit_logo.png" alt="DeepPoseKit logo" height="240">
</p>

# DeepPoseKit

DeepPoseKit trains models for 2D pose estimation of user-defined keypoints and
runs them on images or video. This fork uses Python 3.12, TensorFlow 2.21,
Keras 3, and a locked uv environment. [Original DeepPoseKit paper](https://doi.org/10.7554/eLife.47994)

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
| `[data]` | `DataGenerator` options, including `dataset` and `mode`, with `annotations` supplying the HDF5 path. |
| `[generator]` | `TrainingGenerator` options such as `downsample_factor`, `use_graph`, `graph_scale`, `sigma`, `validation_split`, `shuffle`, and `random_seed`. |
| `[model]` | Constructor options for `StackedDenseNet` (the default), `StackedHourglass`, `LEAP`, or `DeepLabCut`. Set `pretrained_weights = "none"` or `weights = "none"` on models that expose those options to initialize without external weights. |
| `[compile]` | Keras `Model.compile` options. Omit the section for Adam and mean squared error. |
| `[training]` | DeepPoseKit `fit` options and Keras fit controls such as `initial_epoch` and `validation_freq`. Set `batch_size`. Omit `steps_per_epoch` to use all full training batches. |
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

Copy the generated array into `[generator].augmenter`. The transforms run
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
zero-based rows in the annotation HDF5 file.

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
`--skeleton` provides edges for overlays or keypoint names for traces.

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
and preserves reviewed keypoints and review flags.

```bash
deepposekit dataset init diverse.h5 --video video.avi --skeleton skeleton.csv \
  --gray --sample diverse --count 100 --frame-step 10 --seed 17
```

Diversity sampling clusters downsampled images and selects actual source
frames from across clusters. `--frame-step` sets the candidate stride.
`--count` sets the number of frames saved, and `--seed` makes the selection
repeatable.

### Help, completion, and environment

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

# News
- **October 2019:** Our paper describing DeepPoseKit is published at eLife! (http://paper.deepposekit.org)
- **September 2019**: 
    - Nature News covers DeepPoseKit: [Deep learning powers a motion-tracking revolution](http://doi.org/10.1038/d41586-019-02942-5)
    - v0.3.0 is released. See [the release notes](https://github.com/jgraving/DeepPoseKit/releases/tag/v0.3.0).
- **April 2019:** The DeepPoseKit preprint is on biorxiv (http://preprint.deepposekit.org)
