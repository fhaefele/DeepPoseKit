# -*- coding: utf-8 -*-
# Copyright 2018-2019 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from keras import ops
from keras.config import backend, image_data_format

if backend() == "torch":
    from torch.fft import fft2 as _fft2, ifft2 as _ifft2
else:
    from tensorflow.signal import fft2d as _fft2, ifft2d as _ifft2


__all__ = [
    "ndims",
    "fix",
    "fft2d",
    "ifft2d",
    "find_maxima",
    "fftshift1d",
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
]


def normalize_data_format(data_format):
    data_format = (
        image_data_format() if data_format is None else str(data_format).lower()
    )
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown image data format: {data_format!r}")
    return data_format


def ndims(x):
    return ops.ndim(x)


def fix(x):
    return ops.where(x >= 0, ops.floor(x), ops.ceil(x))


def fft2d(x):
    return _fft2(ops.cast(x, "complex64"))


def ifft2d(x):
    return _ifft2(x)


def find_maxima(x):
    col_max = ops.max(x, axis=1)
    row_max = ops.max(x, axis=2)
    cols = ops.cast(ops.argmax(col_max, axis=1), "float32")
    rows = ops.cast(ops.argmax(row_max, axis=1), "float32")
    return ops.stack((rows, cols), axis=-1)


def fftshift1d(x, axis=0):
    """Shift a flattened signal and restore its original shape."""
    x_shape = ops.shape(x)
    flattened = ops.reshape(x, (-1, 1))
    if backend() == "tensorflow":
        from tensorflow.signal import fftshift

        shifted = fftshift(flattened, axes=axis)
    else:
        shifted = ops.roll(flattened, ops.shape(flattened)[axis] // 2, axis=axis)
    return ops.reshape(shifted, x_shape)


def gaussian_kernel_1d(size, sigma):
    sigma = ops.cast(sigma, "float32")
    x = ops.arange(-(size // 2), (size // 2) + 1, dtype="float32")
    kernel = 1 / (sigma * ops.sqrt(ops.cast(2 * np.pi, "float32")))
    kernel = kernel * ops.exp(-0.5 * (x / sigma) ** 2)
    return ops.expand_dims(kernel, axis=-1)


def gaussian_kernel_2d(size, sigma):
    kernel = gaussian_kernel_1d(size, sigma)
    return ops.matmul(kernel, ops.transpose(kernel))
