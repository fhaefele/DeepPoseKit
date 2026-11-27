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

import math

from keras import ops

from deepposekit.models.backend.utils import fft2d, fftshift1d, find_maxima, ifft2d

__all__ = ["_upsampled_registration"]


def _dft_kernel(length, region, upsample_factor, offsets):
    frequency = fftshift1d(ops.arange(length, dtype="float32")) - ops.cast(
        length // 2, "float32"
    )
    samples = ops.arange(region, dtype="float32") - offsets[:, None]
    phase = samples[:, :, None] * frequency
    scale = ops.cast(length * upsample_factor, "complex64")
    return ops.exp((-2j * math.pi / scale) * ops.cast(phase, "complex64"))


def _upsampled_dft(data, region, upsample_factor, offsets):
    """Sample a small upsampled Fourier region without a zero-padded FFT."""
    _, rows, cols = ops.shape(data)
    row_kernel = _dft_kernel(rows, region, upsample_factor, offsets[:, 0])
    col_kernel = _dft_kernel(cols, region, upsample_factor, offsets[:, 1])
    return ops.matmul(
        ops.matmul(row_kernel, data), ops.transpose(col_kernel, (0, 2, 1))
    )


def _upsampled_registration(target_image, src_image, upsample_factor):
    _, rows, cols = ops.shape(src_image)
    src_freq = fft2d(src_image)
    target_freq = fft2d(target_image)

    shape = ops.stack((ops.cast(rows, "float32"), ops.cast(cols, "float32")))
    image_product = src_freq * ops.conj(target_freq)
    cross_correlation = ifft2d(image_product)

    shifts = find_maxima(ops.abs(cross_correlation))
    shifts = ops.where(shifts > ops.floor(shape / 2.0), shifts - shape, shifts)
    shifts = ops.round(shifts * upsample_factor) / upsample_factor

    region = math.ceil(upsample_factor * 1.5)
    dftshift = region // 2
    normalization = ops.cast(rows * cols * upsample_factor**2, "complex64")
    offsets = dftshift - shifts * upsample_factor

    upsampled_dft = _upsampled_dft(
        ops.conj(image_product), region, upsample_factor, offsets
    )
    cross_correlation = ops.abs(ops.conj(upsampled_dft) / normalization)
    maxima = find_maxima(cross_correlation) - dftshift
    return shifts + maxima / upsample_factor
