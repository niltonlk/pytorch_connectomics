"""
Tests for the MedNeXt auto-context (ACLSD / ACRLSD) cascade wrapper.

The real MedNeXt trunk requires the optional ``nnunet_mednext`` dependency, so
the novel cascade logic in ``MedNeXtAutoContextWrapper`` is exercised against a
faithful stub that mirrors the ``MedNeXtMultiHeadWrapper`` seam it relies on:
``forward_features(x) -> features`` plus a ``heads`` ``ModuleDict``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from connectomics.models.architectures import is_architecture_available
from connectomics.models.architectures.mednext_autocontext import (
    MedNeXtAutoContextWrapper,
)


FEATURE_CH = 8
LSD_CHANNELS = 10
AFF_CHANNELS = 6
IN_CHANNELS = 1
SHAPE = (1, IN_CHANNELS, 8, 16, 16)


class StubMultiHeadTrunk(nn.Module):
    """Mimics MedNeXtMultiHeadWrapper: forward_features + named heads."""

    def __init__(self, in_channels, heads):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, FEATURE_CH, kernel_size=3, padding=1)
        self.heads = nn.ModuleDict(
            {name: nn.Conv3d(FEATURE_CH, out_ch, kernel_size=1) for name, out_ch in heads.items()}
        )

    def forward_features(self, x):
        return self.stem(x)

    def forward(self, x):
        f = self.forward_features(x)
        return {"output": {name: head(f) for name, head in self.heads.items()}}


def _wrapper(include_raw=True, detach_stage1=True, freeze_stage1=False):
    stage2_in = LSD_CHANNELS + (IN_CHANNELS if include_raw else 0)
    return MedNeXtAutoContextWrapper(
        StubMultiHeadTrunk(IN_CHANNELS, {"lsd": LSD_CHANNELS}),
        StubMultiHeadTrunk(stage2_in, {"aff": AFF_CHANNELS}),
        include_raw=include_raw,
        detach_stage1=detach_stage1,
        freeze_stage1=freeze_stage1,
    )


def test_registered():
    assert is_architecture_available("mednext_autocontext")


def test_output_contract_and_shapes():
    model = _wrapper()
    out = model(torch.randn(*SHAPE))
    assert set(out["output"].keys()) == {"lsd", "aff"}
    assert out["output"]["lsd"].shape == (1, LSD_CHANNELS) + SHAPE[2:]
    assert out["output"]["aff"].shape == (1, AFF_CHANNELS) + SHAPE[2:]


def test_acrlsd_vs_aclsd_input_width():
    acr = _wrapper(include_raw=True)
    assert acr.stage2.stem.in_channels == LSD_CHANNELS + IN_CHANNELS

    acl = _wrapper(include_raw=False)
    assert acl.stage2.stem.in_channels == LSD_CHANNELS


def test_lsd_output_is_logits_not_sigmoid():
    torch.manual_seed(0)
    model = _wrapper()
    out = model(torch.randn(*SHAPE))
    assert out["output"]["lsd"].min() < 0.0


def _stage1_grad_norm(model, backprop_from="aff"):
    model.zero_grad(set_to_none=True)
    out = model(torch.randn(*SHAPE))
    out["output"][backprop_from].sum().backward()
    grads = [p.grad for p in model.stage1.parameters() if p.grad is not None]
    return float(sum(g.abs().sum() for g in grads)) if grads else 0.0


def test_detach_blocks_affinity_gradients():
    assert _stage1_grad_norm(_wrapper(detach_stage1=True), backprop_from="aff") == 0.0


def test_no_detach_lets_affinity_gradients_reach_stage1():
    assert _stage1_grad_norm(_wrapper(detach_stage1=False), backprop_from="aff") > 0.0


def test_stage1_trainable_from_its_own_loss():
    assert _stage1_grad_norm(_wrapper(detach_stage1=True), backprop_from="lsd") > 0.0


def test_freeze_stage1():
    model = _wrapper(freeze_stage1=True)
    assert all(not p.requires_grad for p in model.stage1.parameters())
    assert model.detach_stage1 is True
    model.train()
    assert not model.stage1.training
    assert model.stage2.training


def test_missing_head_raises():
    with pytest.raises(ValueError, match="lsd"):
        MedNeXtAutoContextWrapper(
            StubMultiHeadTrunk(IN_CHANNELS, {"wrong": LSD_CHANNELS}),
            StubMultiHeadTrunk(LSD_CHANNELS + IN_CHANNELS, {"aff": AFF_CHANNELS}),
        )


def test_missing_forward_features_raises():
    class NoFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleDict({"lsd": nn.Conv3d(1, LSD_CHANNELS, 1)})

        def forward(self, x):  # pragma: no cover - not exercised
            return x

    with pytest.raises(ValueError, match="forward_features"):
        MedNeXtAutoContextWrapper(
            NoFeatures(),
            StubMultiHeadTrunk(LSD_CHANNELS + IN_CHANNELS, {"aff": AFF_CHANNELS}),
        )
