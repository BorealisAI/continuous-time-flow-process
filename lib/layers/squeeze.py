# MIT License
#
# Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Link: https://github.com/rtqichen/ffjord

import torch.nn as nn

__all__ = ["SqueezeLayer"]


class SqueezeLayer(nn.Module):
    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._upsample(x, logpx)
        else:
            return self._downsample(x, logpx)

    def _downsample(self, x, logpx=None):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def _upsample(self, y, logpy=None):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


def unsqueeze(input, upscale_factor=2):
    """
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    """
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor ** 2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width
    )

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    """
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    """
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor ** 2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size,
        in_channels,
        out_height,
        downscale_factor,
        out_width,
        downscale_factor,
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)
