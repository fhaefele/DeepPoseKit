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

import tensorflow as tf
from keras.config import image_data_format
import numpy as np

__all__ = [
    "ndims",
    "fix",
    "fft2d",
    "find_maxima",
    "fftshift1d",
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
]


def normalize_data_format(data_format):
    data_format = image_data_format() if data_format is None else str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown image data format: {data_format!r}")
    return data_format


def ndims(x):
    return tf.size(tf.shape(x))


def fix(x):
    x = tf.where(x >= 0, tf.math.floor(x), tf.math.ceil(x))
    return x


def fft2d(x):
    x = tf.cast(x, tf.complex64)
    x = tf.signal.fft2d(x)
    return x


def find_maxima(x):

    col_max = tf.reduce_max(x, axis=1)
    row_max = tf.reduce_max(x, axis=2)

    cols = tf.cast(tf.argmax(col_max, 1), tf.float32)
    rows = tf.cast(tf.argmax(row_max, 1), tf.float32)
    cols = tf.reshape(cols, (-1, 1))
    rows = tf.reshape(rows, (-1, 1))

    maxima = tf.concat([rows, cols], -1)

    return maxima


def fftshift1d(x, axis=0):
    """Shift a flattened signal and restore its original shape."""
    x_shape = tf.shape(x)
    flattened = tf.reshape(x, (-1, 1))
    shifted = tf.signal.fftshift(flattened, axes=axis)
    return tf.reshape(shifted, x_shape)


def gaussian_kernel_1d(size, sigma):
    size = tf.constant(size, dtype=tf.float32)
    sigma = tf.constant(sigma, dtype=tf.float32)
    x = tf.range(-(size // 2), (size // 2) + 1, dtype=tf.float32)
    kernel = 1 / (sigma * tf.sqrt(2 * np.pi))
    kernel *= tf.exp(-0.5 * (x / sigma) ** 2)
    return tf.expand_dims(kernel, axis=-1)


def gaussian_kernel_2d(size, sigma):
    kernel = gaussian_kernel_1d(size, sigma)
    kernel = tf.matmul(kernel, kernel, transpose_b=True)
    return kernel
