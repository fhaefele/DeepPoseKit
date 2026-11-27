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

from keras import ops
from keras.backend import standardize_dtype
from keras.config import floatx

from deepposekit.models.backend.utils import gaussian_kernel_2d, normalize_data_format
from deepposekit.models.backend.registration import _upsampled_registration

__all__ = [
    "resize_images",
    "find_maxima",
    "find_subpixel_maxima",
    "depth_to_space",
    "space_to_depth",
]


def resize_images(x, height_factor, width_factor, interpolation, data_format):
    """Resizes the images contained in a 4D tensor.
    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        interpolation: string, "nearest", "bilinear" or "bicubic"
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    from torch.nn.functional import interpolate

    data_format = normalize_data_format(data_format)
    x = ops.convert_to_tensor(x)
    if data_format == "channels_last":
        x = ops.transpose(x, (0, 3, 1, 2))
    height, width = x.shape[2:]
    output_size = (height * height_factor, width * width_factor)
    if interpolation == "nearest":
        # Match the original align-corners sampling, including half-up ties.
        for axis, size, scaled in zip((2, 3), (height, width), output_size):
            denominator = max(scaled - 1, 1)
            positions = ops.arange(scaled, dtype="int64") * (size - 1)
            indices = (2 * positions + denominator) // (2 * denominator)
            x = ops.take(x, indices, axis=axis)
    elif interpolation in {"bilinear", "bicubic"}:
        x = interpolate(
            ops.cast(x, "float32"),
            size=output_size,
            mode=interpolation,
            align_corners=True,
        )
    else:
        raise ValueError("Invalid interpolation method:", interpolation)
    if data_format == "channels_last":
        x = ops.transpose(x, (0, 2, 3, 1))
    return x


def _find_maxima(x, coordinate_scale=1, confidence_scale=255.0):

    x = ops.cast(x, floatx())

    col_max = ops.max(x, axis=1)
    row_max = ops.max(x, axis=2)

    maxima = ops.max(col_max, 1)
    maxima = ops.expand_dims(maxima, -2) / confidence_scale

    cols = ops.cast(ops.argmax(col_max, -2), floatx())
    rows = ops.cast(ops.argmax(row_max, -2), floatx())
    cols = ops.expand_dims(cols, -2) * coordinate_scale
    rows = ops.expand_dims(rows, -2) * coordinate_scale

    maxima = ops.concatenate([cols, rows, maxima], -2)

    return maxima


def find_maxima(x, coordinate_scale=1, confidence_scale=255.0, data_format=None):
    """Finds the 2D maxima contained in a 4D tensor.
    # Arguments
        x: Tensor or variable.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == "channels_first":
        x = ops.transpose(x, [0, 2, 3, 1])
        x = _find_maxima(x, coordinate_scale, confidence_scale)
        x = ops.transpose(x, [0, 2, 1])
        return x
    elif data_format == "channels_last":
        x = _find_maxima(x, coordinate_scale, confidence_scale)
        x = ops.transpose(x, [0, 2, 1])
        return x
    else:
        raise ValueError("Invalid data_format:", data_format)


def _find_subpixel_maxima(
    x, kernel_size, sigma, upsample_factor, coordinate_scale=1.0, confidence_scale=1.0
):

    kernel = gaussian_kernel_2d(kernel_size, sigma)
    kernel = ops.expand_dims(kernel, 0)

    x_shape = ops.shape(x)
    rows = x_shape[1]
    cols = x_shape[2]

    max_vals = ops.max(ops.reshape(x, [-1, rows * cols]), axis=1)
    max_vals = ops.reshape(max_vals, [-1, 1]) / confidence_scale

    row_pad = rows // 2 - kernel_size // 2
    col_pad = cols // 2 - kernel_size // 2
    padding = [[0, 0], [row_pad, row_pad - 1], [col_pad, col_pad - 1]]
    kernel = ops.pad(kernel, padding)

    row_center = row_pad + (kernel_size // 2)
    col_center = col_pad + (kernel_size // 2)
    center = ops.stack([row_center, col_center])
    center = ops.expand_dims(center, 0)
    center = ops.cast(center, dtype="float32")

    shifts = _upsampled_registration(x, kernel, upsample_factor)
    shifts = center - shifts
    shifts *= coordinate_scale
    maxima = ops.concatenate([ops.flip(shifts, axis=-1), max_vals], -1)

    return maxima


def find_subpixel_maxima(
    x,
    kernel_size,
    sigma,
    upsample_factor,
    coordinate_scale=1.0,
    confidence_scale=1.0,
    data_format=None,
):
    """Finds the 2D maxima contained in a 4D tensor.
    # Arguments
        x: Tensor or variable.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == "channels_first":
        x = ops.transpose(x, [0, 2, 3, 1])
        x_shape = ops.shape(x)
        batch = x_shape[0]
        row = x_shape[1]
        col = x_shape[2]
        channels = x_shape[3]
        x = ops.transpose(x, [0, 3, 1, 2])
        x = ops.reshape(x, [batch * channels, row, col])
        x = _find_subpixel_maxima(
            x, kernel_size, sigma, upsample_factor, coordinate_scale, confidence_scale
        )
        x = ops.reshape(x, [batch, channels, 3])
        return x
    elif data_format == "channels_last":
        x_shape = ops.shape(x)
        batch = x_shape[0]
        row = x_shape[1]
        col = x_shape[2]
        channels = x_shape[3]
        x = ops.transpose(x, [0, 3, 1, 2])
        x = ops.reshape(x, [batch * channels, row, col])
        x = _find_subpixel_maxima(
            x, kernel_size, sigma, upsample_factor, coordinate_scale, confidence_scale
        )
        x = ops.reshape(x, [batch, channels, 3])

        return x
    else:
        raise ValueError("Invalid data_format:", data_format)


def _preprocess_conv2d_input(x, data_format):
    if standardize_dtype(x.dtype) == "float64":
        x = ops.cast(x, "float32")
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 2, 3, 1))
    return x


def _postprocess_conv2d_output(x, data_format):
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 3, 1, 2))
    if floatx() == "float64":
        x = ops.cast(x, "float64")
    return x


def depth_to_space(input, scale, data_format=None):
    """Move channels into space in the original TensorFlow offset-major order."""
    data_format = normalize_data_format(data_format)
    x = _preprocess_conv2d_input(input, data_format)
    batch, rows, cols, depth = ops.shape(x)
    channels = depth // (scale**2)
    x = ops.reshape(x, (batch, rows, cols, scale, scale, channels))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (batch, rows * scale, cols * scale, channels))
    return _postprocess_conv2d_output(x, data_format)


def space_to_depth(input, scale, data_format=None):
    """Move spatial offsets into channels, reversing depth_to_space."""
    data_format = normalize_data_format(data_format)
    x = _preprocess_conv2d_input(input, data_format)
    batch, rows, cols, channels = ops.shape(x)
    x = ops.reshape(x, (batch, rows // scale, scale, cols // scale, scale, channels))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (batch, rows // scale, cols // scale, channels * scale**2))
    return _postprocess_conv2d_output(x, data_format)
