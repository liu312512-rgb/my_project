# ------------------------------------------------------------------------
# SCI (Self-Calibrated Illumination) — inference-only module
# Extracted from TPAMI 2025: "Learning with Self-Calibrator for Fast
# and Robust Low-Light Image Enhancement"
# ------------------------------------------------------------------------
"""Lightweight low-light image enhancement preprocessor (~250 parameters).

Weights are loaded from the TPAMI checkpoint.  The module expects RGB
tensors in [0, 1] and returns enhanced RGB tensors in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _default_conv(dim_in: int, dim_out: int, kernel_size: int = 3, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size // 2), bias=bias)


class EnhanceNetwork_Ha(nn.Module):
    """Single-block illumination estimator — the only module used at inference.

    Architecture: in_conv → residual conv blocks → out_conv (Sigmoid).
    Input skips around the entire network, so the output is clamped
    illumination in [0.0001, 1].
    """

    def __init__(self, layers: int = 1, channels: int = 3) -> None:
        super().__init__()
        kernel_size = 3#卷积核为3
        self.in_conv = nn.Sequential(
            _default_conv(dim_in=3, dim_out=channels, kernel_size=kernel_size, bias=True),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):#循环，layer次
            conv = nn.Sequential(
                _default_conv(dim_in=channels, dim_out=channels, kernel_size=kernel_size, bias=True),
                nn.ReLU(),
            )
            self.blocks.append(conv)

        self.out_conv = nn.Sequential(
            _default_conv(dim_in=channels, dim_out=3, kernel_size=kernel_size, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + x
        illu = torch.clamp(illu, 0.0001, 1)#把illumination限制在0.0001到1之间  
        return illu


class Finetunemodel(nn.Module):
    """SCI inference wrapper.

    Loads a pretrained checkpoint and exposes a clean forward pass that
    returns (enhanced_image, reflectance).
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        self.ha = EnhanceNetwork_Ha(layers=1, channels=3)
        self._load_weights(weights)

    def _load_weights(self, weights: str) -> None:
        state = torch.load(weights, map_location="cpu")
        # The checkpoint may store a full training model (ha + hb + calibrate)
        # or just the inference model (ha only).  Keep only ha.* keys.
        ha_state = {k: v for k, v in state.items() if k.startswith("ha.")}
        if not ha_state:
            # Fallback: checkpoint was saved from EnhanceNetwork_Ha directly
            # (no "ha." prefix).  Load as-is.
            ha_state = state
        self.load_state_dict(ha_state, strict=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhance a low-light image.

        Args:
            x: RGB tensor of shape ``(..., C, H, W)`` in [0, 1].

        Returns:
            ``(enhanced, reflectance)`` where both are in [0, 1].
        """
        i = self.ha(x)
        r = x / i
        r = torch.clamp(r, 0, 1)
        return i, r
