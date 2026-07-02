"""
Generic auto-context (ACLSD / ACRLSD) cascade for connectomics.

Implements the auto-context Local Shape Descriptor idea from Sheridan et al.
(*Local shape descriptors for neuron segmentation*, Nature Methods 2022) as a
single end-to-end two-stage cascade that plugs into the existing Lightning
training loop with zero changes to the loss system:

    stage 1:  raw               -> LSD logits
    bridge:   sigmoid(LSD)      -> [0, 1] feed (matches the reference bounded LSD)
    stage 2:  [LSD_feed (+raw)] -> affinity logits

The wrapper returns the standard named-head contract

    {"output": {"lsd": lsd_logits, "aff": aff_logits}}

so ``LossOrchestrator`` can route ``pred_head: lsd`` and ``pred_head: aff`` loss
terms unchanged. The LSD tensor handed to the loss is **raw logits**
(``*WithLogits`` friendly); only the bridge into stage 2 is sigmoided.

Unlike ``mednext_autocontext.MedNeXtAutoContextWrapper`` (which relies on the
MedNeXt-specific ``forward_features`` / ``heads`` seam), this cascade treats each
stage as a black box: it calls ``stage(x)`` and reads the resulting tensor via
``_extract_main_tensor``. That makes it work with RSUNet, MONAI, or a single-head
MedNeXt trunk. The ``rsunet_aclsd`` builder wires two RSUNet stages together.

Gradient modes (config ``model.arch.params``):

- ``detach_stage1: true`` (default) -- decouple the stages in a single run
  (paper-like: stage 2 does not shape the LSDs).
- ``detach_stage1: false`` -- affinity gradients also flow back into the LSDs.
- ``freeze_stage1: true`` -- literal paper setup: stage 1 is frozen (implies
  detach) and kept in eval mode so its norm statistics never update. Combine
  with ``stage1_weights_path`` to load a pre-trained LSD network.
- ``include_raw`` picks ACRLSD (``true``, raw concatenated with the LSD feed)
  vs ACLSD (``false``, LSD feed only).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch

from .base import ConnectomicsModel
from .registry import register_architecture
from .rsunet import RSUNet

logger = logging.getLogger(__name__)


class AutoContextCascade(ConnectomicsModel):
    """Generic two-stage auto-context cascade (ACLSD / ACRLSD).

    Args:
        stage1: LSD network. Called as ``stage1(raw)``; its output is unwrapped
            to a single tensor of shape ``(B, lsd_channels, ...)``.
        stage2: affinity network. Called as ``stage2(stage2_input)``; its output
            is unwrapped to a single tensor of shape ``(B, aff_channels, ...)``.
        include_raw: ``True`` -> ACRLSD (concatenate raw with the sigmoided LSD
            feed for stage 2). ``False`` -> ACLSD (LSD feed only).
        detach_stage1: detach the LSD feed before stage 2 so affinity gradients
            do not flow into stage 1.
        freeze_stage1: freeze stage 1 parameters and keep it in eval mode.
            Implies ``detach_stage1``.
        lsd_head: output-head name used for the LSD logits.
        aff_head: output-head name used for the affinity logits.
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
        self.stage1 = stage1
        self.stage2 = stage2
        self.include_raw = include_raw
        self.freeze_stage1 = freeze_stage1
        # A frozen stage 1 must not receive gradients from stage 2 either.
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
    def _extract_main_tensor(out: Any) -> torch.Tensor:
        """Unwrap a stage output to its single main tensor.

        Accepts a bare tensor, ``{"output": tensor}``, ``{"output": {head:
        tensor}}`` with a single head, or a single-head bare mapping. Rejects an
        ambiguous multi-head mapping, since the cascade needs one unambiguous
        tensor per stage.
        """
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, Mapping):
            if "output" in out:
                inner = out["output"]
                if isinstance(inner, torch.Tensor):
                    return inner
                if isinstance(inner, Mapping):
                    if len(inner) == 1:
                        value = next(iter(inner.values()))
                        if isinstance(value, torch.Tensor):
                            return value
                    raise ValueError(
                        "AutoContextCascade cannot extract a single tensor from a "
                        f"multi-head stage output with heads {sorted(inner.keys())}. "
                        "Give the stage exactly one head."
                    )
                raise TypeError(
                    f"Unexpected stage 'output' value of type {type(inner).__name__}."
                )
            if len(out) == 1:
                value = next(iter(out.values()))
                if isinstance(value, torch.Tensor):
                    return value
            raise ValueError(
                "AutoContextCascade cannot extract a single tensor from stage output "
                f"with keys {sorted(out.keys())}. Expected a tensor, {{'output': "
                "tensor}}, or a single-head mapping."
            )
        raise TypeError(
            f"Stage output must be a tensor or mapping, got {type(out).__name__}."
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        lsd_logits = self._extract_main_tensor(self.stage1(x))

        # LSDs are bounded in [0, 1]; feed the bounded value into stage 2 to
        # match the reference auto-context input, but supervise raw logits.
        lsd_feed = torch.sigmoid(lsd_logits)
        if self.detach_stage1:
            lsd_feed = lsd_feed.detach()

        stage2_input = torch.cat([lsd_feed, x], dim=1) if self.include_raw else lsd_feed
        aff_logits = self._extract_main_tensor(self.stage2(stage2_input))

        return {"output": {self.lsd_head: lsd_logits, self.aff_head: aff_logits}}

    def train(self, mode: bool = True) -> "AutoContextCascade":
        """Keep a frozen stage 1 in eval mode regardless of the parent switch."""
        super().train(mode)
        if self.freeze_stage1:
            self.stage1.eval()
        return self


# ============================================================================
# RSUNet auto-context builder
# ============================================================================


def _as_plain_dict(value: Any) -> Dict[str, Any]:
    """Normalize an OmegaConf/dict-like value into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    # OmegaConf DictConfig supports item access but is not a Mapping subclass in
    # every version; fall back to its container form when available.
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True))
    except Exception:  # pragma: no cover - omegaconf always present in repo
        pass
    return dict(value)


def _rsunet_from_cfg(
    cfg,
    *,
    in_channels: int,
    out_channels: int,
    overrides: Optional[Mapping[str, Any]] = None,
) -> RSUNet:
    """Build one RSUNet stage from ``cfg.model.rsunet`` plus per-stage overrides."""
    base = cfg.model.rsunet
    overrides = _as_plain_dict(overrides)

    def _get(key: str, default: Any) -> Any:
        if key in overrides:
            return overrides[key]
        return getattr(base, key, default)

    width = list(_get("width", [16, 32, 64, 128]))

    down_factors = _get("down_factors", None)
    if down_factors is not None:
        down_factors = [tuple(f) for f in down_factors]

    kernel_2d = tuple(_get("kernel_2d", (1, 3, 3)))

    act_kwargs = {
        "negative_slope": _get("act_negative_slope", 0.01),
        "init": _get("act_init", 0.25),
    }

    return RSUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        width=width,
        norm=_get("norm", "batch"),
        activation=_get("activation", "relu"),
        num_groups=_get("num_groups", 8),
        deep_supervision=False,  # cascade stages emit a single tensor
        down_factors=down_factors,
        depth_2d=_get("depth_2d", 0),
        kernel_2d=kernel_2d,
        **act_kwargs,
    )


def _maybe_load_stage1_weights(stage1: torch.nn.Module, path: Optional[str], prefix: str) -> None:
    """Load a pre-trained LSD network into stage 1 (best-effort, strict=False)."""
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
    if prefix:
        state = {
            (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()
        }
    missing, unexpected = stage1.load_state_dict(state, strict=False)
    logger.info(
        "Loaded stage-1 LSD weights from %s (missing=%d, unexpected=%d)",
        path,
        len(missing),
        len(unexpected),
    )


@register_architecture("rsunet_aclsd")
def build_rsunet_aclsd(cfg) -> AutoContextCascade:
    """Build an RSUNet auto-context LSD cascade (ACLSD / ACRLSD).

    Reads cascade options from ``cfg.model.arch.params`` (a free-form dict):

        model:
          arch:
            type: rsunet_aclsd
            params:
              lsd_channels: 10      # 3D LSD width (6 for 2D)
              aff_channels: 6       # affinity head width
              include_raw: true     # true -> ACRLSD, false -> ACLSD
              detach_stage1: true   # decouple the two stages in one run
              freeze_stage1: false  # freeze + eval stage 1 (paper setup)
              stage1_weights_path: null
              stage1: {}            # per-stage RSUNet overrides
              stage2: {}
          in_channels: 1
          out_channels: 6           # = aff_channels
          rsunet:
            width: [16, 32, 64, 128]
            norm: group

    Shared RSUNet knobs come from ``cfg.model.rsunet``; ``params.stage1`` /
    ``params.stage2`` override them per stage.
    """
    params = _as_plain_dict(getattr(cfg.model.arch, "params", None))

    lsd_channels = int(params.get("lsd_channels", 10))
    aff_channels = int(params.get("aff_channels", cfg.model.out_channels))
    include_raw = bool(params.get("include_raw", True))
    detach_stage1 = bool(params.get("detach_stage1", True))
    freeze_stage1 = bool(params.get("freeze_stage1", False))
    lsd_head = str(params.get("lsd_head", "lsd"))
    aff_head = str(params.get("aff_head", "aff"))

    in_channels = int(cfg.model.in_channels)

    stage1 = _rsunet_from_cfg(
        cfg,
        in_channels=in_channels,
        out_channels=lsd_channels,
        overrides=params.get("stage1"),
    )

    stage2_in = lsd_channels + (in_channels if include_raw else 0)
    stage2 = _rsunet_from_cfg(
        cfg,
        in_channels=stage2_in,
        out_channels=aff_channels,
        overrides=params.get("stage2"),
    )

    _maybe_load_stage1_weights(
        stage1,
        params.get("stage1_weights_path") or getattr(cfg.model, "external_weights_path", None),
        str(params.get("stage1_weights_key_prefix", getattr(cfg.model, "external_weights_key_prefix", ""))),
    )

    return AutoContextCascade(
        stage1,
        stage2,
        include_raw=include_raw,
        detach_stage1=detach_stage1,
        freeze_stage1=freeze_stage1,
        lsd_head=lsd_head,
        aff_head=aff_head,
    )


__all__ = ["AutoContextCascade", "build_rsunet_aclsd"]
