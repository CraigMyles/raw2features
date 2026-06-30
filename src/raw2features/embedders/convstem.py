"""ConvStem patch-embed for CTransPath (Wang et al., *TransPath*).

CTransPath replaces a Swin-Tiny's single-conv patch embed with a small multi-layer
conv stem. This module reproduces that architecture so the published checkpoint loads
via ``timm.create_model(..., embed_layer=ConvStem)`` on **modern** timm - no pinned
timm fork, no ancient Python.

The layer structure is dictated by the checkpoint (it must match exactly to load), so
it is re-implemented here rather than copied. Architecture reference:
https://github.com/Xiyue-Wang/TransPath (``ctran.py``), via the modern-timm mirror
``1aurent/swin_tiny_patch4_window7_224.CTransPath``.

NOTE: the CTransPath **weights** are GPL-3.0 - see ``docs/MODEL_LICENSES.md``.
This loader module is part of raw2features (MIT); using GPL weights at runtime does not
relicense it.
"""

from __future__ import annotations

import torch.nn as nn
from timm.layers.helpers import to_2tuple


class ConvStem(nn.Module):
    """Multi-conv patch-embed stem: 3 -> embed_dim/8 -> embed_dim/4 -> embed_dim.

    timm passes the host model's ``embed_dim`` (96 for Swin-Tiny), so the stem builds
    3->12->24->96 with two stride-2 conv blocks (4x downsample = patch_size 4) and a
    final 1x1 conv. Returns ``BHWC`` (what modern timm's Swin patch-embed slot expects).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert patch_size == 4, "ConvStem assumes patch_size == 4"
        assert embed_dim % 8 == 0, "embed_dim must be divisible by 8"
        img_size, patch_size = to_2tuple(img_size), to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        layers: list[nn.Module] = []
        c_in, c_out = in_chans, embed_dim // 8
        for _ in range(2):
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in, c_out = c_out, c_out * 2
        layers.append(nn.Conv2d(c_in, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*layers)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return self.norm(x)
