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

Use the `gpu` extra on x86-64 Linux or WSL2 with a working NVIDIA driver.
The `deepposekit-gpu` command runs Python with the CUDA libraries and compiler
from the locked environment. [TensorFlow GPU installation guide](https://www.tensorflow.org/install/pip)

Save this as `check_gpu.py`.

```python
import tensorflow as tf
from deepposekit.models import StackedDenseNet

devices = tf.config.list_physical_devices("GPU")
assert devices, "No GPU detected"
print(devices)
```

```bash
nvidia-smi
uv sync --locked --extra gpu
uv run --locked --extra gpu deepposekit-gpu check_gpu.py
```

Use `uv run --locked --extra gpu deepposekit-gpu your_script.py` for the training
and prediction examples below. GPU training and legacy subpixel prediction were
tested on an RTX 4090 in WSL2.

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

For GPU training or prediction, use the launcher with the `gpu` extra.

```bash
uv run --locked --extra gpu deepposekit-gpu -m deepposekit train training.toml
uvx --python 3.12 --from "deepposekit[gpu] @ $fork" deepposekit-gpu -m deepposekit train training.toml
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

For fine-tuning, use compatible labeled annotations and a new output path in
`fine-tune.toml`. Set `[model].name` for other backbones. Omit architecture
and confidence map settings to inherit them from the checkpoint. Training uses
a fresh optimizer. For older checkpoints with incomplete skeleton metadata,
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

A custom reader can open recordings that the built-in video reader cannot.

```bash
deepposekit predict model.keras recording.h5 predictions.npy \
  --batch-size 8 --custom-reader ./reader.py \
  --custom-reader-args --camera Camera1 --gray --uint8
```

DeepPoseKit loads the selected local Python file and calls
`create_reader(path, *, batch_size, argv)`. The factory gets a `Path`, the
requested batch size, and a list of every argument after
`--custom-reader-args`. Put DeepPoseKit flags before that marker. Return an
iterable of NumPy arrays shaped `(batch, height, width, channels)` in frame
order, including the final short batch. DeepPoseKit closes the iterator and
reader when they provide `close()` methods. Prediction takes one recording
per invocation.

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
dataset. Run your script with `uv run --locked python your_script.py` for
the base install or `uv run --locked --extra gpu deepposekit-gpu your_script.py` on GPU.
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
