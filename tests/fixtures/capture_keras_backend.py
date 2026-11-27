"""Capture matched DeepPoseKit inference for a Keras backend experiment."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import h5py
import keras
import numpy as np


def gpu():
    if keras.config.backend() == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is required for this experiment")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.no_grad(), torch.cuda.get_device_name(0), "cuda"

    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("A TensorFlow GPU is required for this experiment")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    name = tf.config.experimental.get_device_details(devices[0]).get("device_name")
    return tf.device("/GPU:0"), name, "GPU"


def capture(args):
    from deepposekit.models import load_model

    context, device, device_token = gpu()
    args.output.mkdir()
    with context, h5py.File(args.annotations, "r") as annotations:
        images = annotations["images"]
        count = min(args.frames or len(images), len(images))
        pose = load_model(args.checkpoint, compile=False)
        heatmaps = pose.train_model(pose.inputs)
        if isinstance(heatmaps, (list, tuple)):
            heatmaps = heatmaps[-1]
        coordinates = pose.predict_model.layers[-1](heatmaps)
        model = keras.Model(pose.inputs, (heatmaps, coordinates))
        warmup = model(images[: min(count, args.batch_size)], training=False)
        if any(device_token not in str(output.device) for output in warmup):
            raise RuntimeError(
                f"Model outputs did not stay on the GPU: {[x.device for x in warmup]}"
            )
        arrays = [keras.ops.convert_to_numpy(value) for value in warmup]
        destinations = [
            np.lib.format.open_memmap(
                args.output / f"{name}.npy",
                mode="w+",
                dtype=array.dtype,
                shape=(count, *array.shape[1:]),
            )
            for name, array in zip(("heatmaps", "predictions"), arrays, strict=True)
        ]
        latencies = []
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            frames = images[start:stop]
            started = perf_counter()
            outputs = model(frames, training=False)
            values = [keras.ops.convert_to_numpy(value) for value in outputs]
            latencies.append(perf_counter() - started)
            for output, destination in zip(values, destinations, strict=True):
                if not np.isfinite(output).all():
                    raise RuntimeError(f"Non-finite output at frames {start}:{stop}")
                destination[start:stop] = output

        for destination in destinations:
            destination.flush()

    backend = keras.config.backend()
    summary = {
        "backend": backend,
        "backend_version": version(backend),
        "keras": keras.__version__,
        "gpu": device,
        "frames": count,
        "batch_size": args.batch_size,
        "median_full_batch_ms": 1000
        * float(np.median(latencies[: count // args.batch_size] or latencies)),
        "total_inference_seconds": sum(latencies),
        "tensorflow_loaded": "tensorflow" in sys.modules,
    }
    (args.output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.frames is not None and args.frames < 1):
        parser.error("batch size and frame limit must be positive")
    capture(args)


if __name__ == "__main__":
    main()
