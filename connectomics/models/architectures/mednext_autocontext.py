"""
MedNeXt auto-context (ACLSD / ACRLSD) cascade.

A MedNeXt-native variant of the auto-context Local Shape Descriptor cascade
(Sheridan et al., Nature Methods 2022). Unlike the generic
``autocontext.AutoContextCascade`` (which treats each stage as an opaque
``stage(x) -> tensor``), this wrapper uses the MedNeXt-specific seam exposed by
``MedNeXtMultiHeadWrapper``:

    features = stage.forward_features(x)
    logits   = stage.heads[name](features)

so it can reuse the shared MedNeXt trunk feature map and its typed task heads.

Forward pass:

    f1        = stage1.forward_features(raw)
    lsd_logits = stage1.heads["lsd"](f1)          # supervised as raw logits
    lsd_feed  = sigmoid(lsd_logits)               # LSDs are [0, 1]
    (detach lsd_feed if detach_stage1)
    x2        = cat([lsd_feed, raw]) if include_raw else lsd_feed   # ACRLSD / ACLSD
    f2        = stage2.forward_features(x2)
    aff_logits = stage2.heads["aff"](f2)
    return {"output": {"lsd": lsd_logits, "aff": aff_logits}}

Returns the standard named-head contract so the loss orchestrator can route
``pred_head: lsd`` / ``pred_head: aff`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import torch

from .base import ConnectomicsModel
from .registry import register_architecture

logger = logging.getLogger(__name__)


class MedNeXtAutoContextWrapper(ConnectomicsModel):
    """Two-stage MedNeXt auto-context cascade (ACLSD / ACRLSD).

    Args:
        stage1: a ``MedNeXtMultiHeadWrapper`` (or any module exposing
            ``forward_features`` and a ``heads`` ``ModuleDict``) that carries the
            LSD head.
        stage2: same contract, carrying the affinity head.
        include_raw: ``True`` -> ACRLSD, ``False`` -> ACLSD.
        detach_stage1: detach the sigmoided LSD feed before stage 2.
        freeze_stage1: freeze stage 1 and keep it in eval mode (implies detach).
            MedNeXt uses GroupNorm, so eval mode is not strictly required for
            correctness, but it keeps behavior identical to the paper.
        lsd_head: head name on stage 1 producing LSD logits.
        aff_head: head name on stage 2 producing affinity logits.
    """

    def __init__(
        self,
        stage1: torch.nn.Module,
        stage2: torch.nn.Module,
        *,
        include_raw: bool = True,
        detach_stage1: bool = True,
        freeze_stage1: bool = False,
        lsd_head: str = "lsd",
        aff_head: str = "aff",
    ):
        super().__init__()
        self._validate_stage(stage1, lsd_head, "stage1")
        self._validate_stage(stage2, aff_head, "stage2")

        self.stage1 = stage1
        self.stage2 = stage2
        self.include_raw = include_raw
        self.freeze_stage1 = freeze_stage1
        self.detach_stage1 = detach_stage1 or freeze_stage1
        self.lsd_head = lsd_head
        self.aff_head = aff_head
        self.supports_deep_supervision = False
        self.output_scales = 1

        if freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad = False
            self.stage1.eval()

    @staticmethod
    def _validate_stage(stage: torch.nn.Module, head: str, label: str) -> None:
        if not hasattr(stage, "forward_features"):
            raise ValueError(
                f"MedNeXtAutoContextWrapper {label} must expose forward_features(); "
                f"got {type(stage).__name__}."
            )
        heads = getattr(stage, "heads", None)
        if heads is None or head not in heads:
            available = sorted(heads.keys()) if heads is not None else []
            raise ValueError(
                f"MedNeXtAutoContextWrapper {label} must expose a '{head}' head; "
                f"available heads: {available}."
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        f1 = self.stage1.forward_features(x)
        lsd_logits = self.stage1.heads[self.lsd_head](f1)

        lsd_feed = torch.sigmoid(lsd_logits)
        if self.detach_stage1:
            lsd_feed = lsd_feed.detach()

        stage2_input = torch.cat([lsd_feed, x], dim=1) if self.include_raw else lsd_feed
        f2 = self.stage2.forward_features(stage2_input)
        aff_logits = self.stage2.heads[self.aff_head](f2)

        return {"output": {self.lsd_head: lsd_logits, self.aff_head: aff_logits}}

    def train(self, mode: bool = True) -> "MedNeXtAutoContextWrapper":
        super().train(mode)
        if self.freeze_stage1:
            self.stage1.eval()
        return self


# ============================================================================
# Builder
# ============================================================================


def _as_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True))
    except Exception:  # pragma: no cover
        pass
    return dict(value)


def _build_mednext_multihead_stage(
    *,
    in_channels: int,
    head_name: str,
    head_channels: int,
    head_num_blocks: int,
    hidden_channels: Optional[int],
    model_size: str,
    kernel_size: int,
    checkpoint_style: Optional[str],
):
    """Build a single-head ``MedNeXtMultiHeadWrapper`` trunk for one cascade stage."""
    from .mednext_models import (
        MedNeXtMultiHeadWrapper,
        _check_mednext_available,
        create_mednext_v1,
    )

    _check_mednext_available()

    trunk = create_mednext_v1(
        num_input_channels=in_channels,
        num_classes=max(1, head_channels),
        model_id=model_size,
        kernel_size=kernel_size,
        deep_supervision=False,
    )
    if checkpoint_style is not None:
        if checkpoint_style != "outside_block":
            raise ValueError(
                "mednext checkpoint_style must be None or 'outside_block', "
                f"got: {checkpoint_style!r}"
            )
        trunk.outside_block_checkpointing = True

    heads = {
        head_name: {
            "out_channels": head_channels,
            "num_blocks": head_num_blocks,
            "hidden_channels": hidden_channels,
        }
    }
    return MedNeXtMultiHeadWrapper(trunk, heads, primary_head=head_name)


@register_architecture("mednext_autocontext")
def build_mednext_autocontext(cfg) -> MedNeXtAutoContextWrapper:
    """Build a MedNeXt auto-context LSD cascade (ACLSD / ACRLSD).

    Reads cascade options from ``cfg.model.arch.params``:

        model:
          arch:
            type: mednext_autocontext
            params:
              lsd_channels: 10
              aff_channels: 6
              include_raw: true       # true -> ACRLSD, false -> ACLSD
              detach_stage1: true
              freeze_stage1: false
              head_num_blocks: 1
              hidden_channels: 16
          in_channels: 1
          out_channels: 6
          mednext:
            size: L
            kernel_size: 3

    Shared MedNeXt knobs (size, kernel_size, checkpoint_style) come from
    ``cfg.model.mednext`` and are used for both stages.
    """
    params = _as_plain_dict(getattr(cfg.model.arch, "params", None))

    lsd_channels = int(params.get("lsd_channels", 10))
    aff_channels = int(params.get("aff_channels", cfg.model.out_channels))
    include_raw = bool(params.get("include_raw", True))
    detach_stage1 = bool(params.get("detach_stage1", True))
    freeze_stage1 = bool(params.get("freeze_stage1", False))
    head_num_blocks = int(params.get("head_num_blocks", 1))
    hidden_channels = params.get("hidden_channels", None)
    hidden_channels = int(hidden_channels) if hidden_channels is not None else None
    lsd_head = str(params.get("lsd_head", "lsd"))
    aff_head = str(params.get("aff_head", "aff"))

    in_channels = int(cfg.model.in_channels)
    model_size = getattr(cfg.model.mednext, "size", "S")
    kernel_size = getattr(cfg.model.mednext, "kernel_size", 3)
    checkpoint_style = getattr(cfg.model.mednext, "checkpoint_style", None)

    stage1 = _build_mednext_multihead_stage(
        in_channels=in_channels,
        head_name=lsd_head,
        head_channels=lsd_channels,
        head_num_blocks=head_num_blocks,
        hidden_channels=hidden_channels,
        model_size=model_size,
        kernel_size=kernel_size,
        checkpoint_style=checkpoint_style,
    )

    stage2_in = lsd_channels + (in_channels if include_raw else 0)
    stage2 = _build_mednext_multihead_stage(
        in_channels=stage2_in,
        head_name=aff_head,
        head_channels=aff_channels,
        head_num_blocks=head_num_blocks,
        hidden_channels=hidden_channels,
        model_size=model_size,
        kernel_size=kernel_size,
        checkpoint_style=checkpoint_style,
    )

    return MedNeXtAutoContextWrapper(
        stage1,
        stage2,
        include_raw=include_raw,
        detach_stage1=detach_stage1,
        freeze_stage1=freeze_stage1,
        lsd_head=lsd_head,
        aff_head=aff_head,
    )


__all__ = ["MedNeXtAutoContextWrapper", "build_mednext_autocontext"]
