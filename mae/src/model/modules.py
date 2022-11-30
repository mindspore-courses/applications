# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""
Modules.
"""

from mindspore import nn
from mindspore import ops as P
from mindspore.common.initializer import initializer, XavierUniform

# Set the minimum number of patches
MIN_NUM_PATCHES = 4

class BatchDense(nn.Cell):
    """
    BatchDense definition.

    Args:
        in_features (int): Features of input.
        out_features (int): Features of output.
        initialization (initializer): Initialization method.
        has_bias (bool): Whether bias.

    Returns:
        Tensor, output tensor, the shape is (batch_size * 3).
    """
    def __init__(self, in_features, out_features, initialization, has_bias=True):
        super(BatchDense, self).__init__()
        self.out_features = out_features
        self.dense = nn.Dense(in_features, out_features, has_bias=has_bias)
        self.dense.weight.set_data(initializer(initialization, [out_features, in_features]))
        self.reshape = P.Reshape()
        self.pixel_values = self.dense.weight.shape[-1]

    def construct(self, x):
        """ build network """

        # Get the three parameters batch_size, seq_len and dim
        bs, seq_len, dim = x.shape

        # By reshape into the fully connected layer and then reshape to the original
        out = self.reshape(x, (bs * seq_len, dim))
        out = self.dense(out)
        out = self.reshape(out, (bs, seq_len, self.out_features))
        return out


class VitStem(nn.Cell):
    """
    ViT Stem layer

    Args:
        dim (int): Number of dimension.
        patch_size (int): Patch size.
        image_size (int): Image size. Default: 3.
        channels (int): Number of channel.
        initialization (initializer): Initialization method.

    Returns:
        Tensor, the Embedding of patch.
        int, Number of patch.
    """
    def __init__(self, dim, patch_size, image_size, channels=3, initialization=XavierUniform()):
        super(VitStem, self).__init__()

        # Determine if the size of the image is divisible by patch_size.
        if image_size % patch_size != 0:
            print('Image dimensions must be divisible by the patch size.')

        # Get the number after conversion to patch.
        num_patches = (image_size // patch_size) ** 2
        if num_patches <= MIN_NUM_PATCHES:
            print(f'your number of patches {num_patches} is too small')

        # Calculate the corresponding dimensional transformation after conversion to patch.
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.patch_to_embedding = BatchDense(patch_dim, dim, initialization, has_bias=True)

    def construct(self, img):
        """ build network """
        p = self.patch_size

        # get batch_size, channels, height, width
        bs, channels, h, w = img.shape

        # reshape to other formats
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        patches = self.reshape(x, (bs, (h//p)*(w//p), channels*p*p))

        # Embedding of patches through a fully connected layer
        x = self.patch_to_embedding(patches)
        return x, patches


class Patchify(nn.Cell):
    """
    Convert images to patches

    Args:
        patch_size (int): Patch size.

    Returns:
        int, Number of patch.
    """

    def __init__(self, patch_size):
        super(Patchify, self).__init__()

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, img):
        """ build network """
        p = self.patch_size
        bs, channels, h, w = img.shape
        x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))

        # Convert image data to patch format
        patches = self.reshape(x, (bs, (h//p)*(w//p), channels*p*p))
        return patches


class PatchEmbed(nn.Cell):
    """
    Create positional embedding for patch positional encoding.

    Args:
        img_size (int): image size. Default: 224.
        patch_size (int): patch size. Default: 16.
        in_features (int): input features. Default: 3.
        out_features (int): output features. Default: 768.

    Returns:
        Tensor, the positional encoding.
    """

    def __init__(self, img_size=224, patch_size=16, in_features=3, out_features=768):
        super(PatchEmbed, self).__init__()
        self.hybrid = None
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        # Calculate the shape (length and width) of the patch
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.projection = nn.Conv2d(in_channels=in_features,
                                    out_channels=out_features,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    has_bias=True)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        """ build network """
        x = self.projection(x)
        x = self.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.transpose(x, (0, 2, 1))
        return x
