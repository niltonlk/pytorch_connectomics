"""
Tests for the generic auto-context (ACLSD / ACRLSD) cascade.

RSUNet is pure PyTorch, so the cascade is exercised end-to-end against real
RSUNet stages and the full ``rsunet_aclsd`` builder path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from connectomics.models.architectures import (
    get_architecture_builder,
    is_architecture_available,
)
from connectomics.models.architectures.autocontext import (
    AutoContextCascade,
    build_rsunet_aclsd,
)
from connectomics.models.architectures.rsunet import RSUNet


LSD_CHANNELS = 10
AFF_CHANNELS = 6
IN_CHANNELS = 1
SHAPE = (1, IN_CHANNELS, 8, 16, 16)


def _rsunet(in_ch, out_ch):
    # Small, GroupNorm stages so a batch size of 1 works and freezing is clean.
    return RSUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        width=[8, 16],
        norm="group",
        num_groups=4,
    )


def _cascade(include_raw=True, detach_stage1=True, freeze_stage1=False):
    stage2_in = LSD_CHANNELS + (IN_CHANNELS if include_raw else 0)
    return AutoContextCascade(
        _rsunet(IN_CHANNELS, LSD_CHANNELS),
        _rsunet(stage2_in, AFF_CHANNELS),
        include_raw=include_raw,
        detach_stage1=detach_stage1,
        freeze_stage1=freeze_stage1,
    )


# ---------------------------------------------------------------------------
# Output contract & shapes
# ---------------------------------------------------------------------------


def test_output_contract_and_shapes():
    model = _cascade()
    out = model(torch.randn(*SHAPE))

    assert set(out.keys()) == {"output"}
    heads = out["output"]
    assert set(heads.keys()) == {"lsd", "aff"}
    assert heads["lsd"].shape == (1, LSD_CHANNELS) + SHAPE[2:]
    assert heads["aff"].shape == (1, AFF_CHANNELS) + SHAPE[2:]


def test_acrlsd_vs_aclsd_input_width():
    # ACRLSD: stage 2 sees LSD feed + raw.
    acr = _cascade(include_raw=True)
    assert acr.stage2.input_conv.pre[1].in_channels == LSD_CHANNELS + IN_CHANNELS

    # ACLSD: stage 2 sees the LSD feed only.
    acl = _cascade(include_raw=False)
    assert acl.stage2.input_conv.pre[1].in_channels == LSD_CHANNELS


def test_lsd_output_is_logits_not_sigmoid():
    # The lsd head returned for the loss must be raw logits (can be negative),
    # even though the bridge into stage 2 is sigmoided.
    torch.manual_seed(0)
    model = _cascade()
    out = model(torch.randn(*SHAPE))
    lsd = out["output"]["lsd"]
    assert lsd.min() < 0.0, "LSD output should be raw logits, not a sigmoid in [0, 1]"


# ---------------------------------------------------------------------------
# Gradient modes
# ---------------------------------------------------------------------------


def _stage1_grad_norm(model, backprop_from="aff"):
    model.zero_grad(set_to_none=True)
    out = model(torch.randn(*SHAPE))
    out["output"][backprop_from].sum().backward()
    grads = [p.grad for p in model.stage1.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(sum(g.abs().sum() for g in grads))


def test_detach_blocks_affinity_gradients_into_stage1():
    model = _cascade(detach_stage1=True)
    # Affinity loss must not reach stage 1 when detached.
    assert _stage1_grad_norm(model, backprop_from="aff") == 0.0


def test_no_detach_lets_affinity_gradients_reach_stage1():
    model = _cascade(detach_stage1=False)
    assert _stage1_grad_norm(model, backprop_from="aff") > 0.0


def test_stage1_trainable_from_its_own_lsd_loss():
    # Even when detached from stage 2, stage 1 must learn from its own LSD loss.
    model = _cascade(detach_stage1=True)
    assert _stage1_grad_norm(model, backprop_from="lsd") > 0.0


def test_freeze_stage1_disables_grad_and_eval():
    model = _cascade(freeze_stage1=True)
    assert all(not p.requires_grad for p in model.stage1.parameters())
    assert model.detach_stage1 is True

    # Frozen stage 1 stays in eval mode even after .train().
    model.train()
    assert not model.stage1.training
    assert model.stage2.training

    # Stage 2 still learns.
    assert _stage1_grad_norm(model, backprop_from="aff") == 0.0
    model.zero_grad(set_to_none=True)
    out = model(torch.randn(*SHAPE))
    out["output"]["aff"].sum().backward()
    assert any(p.grad is not None for p in model.stage2.parameters())


# ---------------------------------------------------------------------------
# _extract_main_tensor
# ---------------------------------------------------------------------------


def test_extract_main_tensor_variants():
    t = torch.randn(1, 3, 4, 4, 4)
    ex = AutoContextCascade._extract_main_tensor

    assert ex(t) is t
    assert ex({"output": t}) is t
    assert ex({"output": {"only": t}}) is t
    assert ex({"only": t}) is t


def test_extract_main_tensor_rejects_ambiguous_multihead():
    a = torch.randn(1, 3, 4, 4, 4)
    b = torch.randn(1, 3, 4, 4, 4)
    with pytest.raises(ValueError):
        AutoContextCascade._extract_main_tensor({"output": {"aff": a, "lsd": b}})
    with pytest.raises(ValueError):
        AutoContextCascade._extract_main_tensor({"aff": a, "lsd": b})


def test_extract_main_tensor_rejects_bad_type():
    with pytest.raises(TypeError):
        AutoContextCascade._extract_main_tensor(42)


def test_cascade_works_with_single_head_dict_stage():
    """A stage that returns a bare {'output': {head: tensor}} still unwraps."""

    class SingleHeadStage(nn.Module):
        def __init__(self, in_ch, out_ch, head):
            super().__init__()
            self.conv = nn.Conv3d(in_ch, out_ch, 1)
            self.head = head

        def forward(self, x):
            return {"output": {self.head: self.conv(x)}}

    model = AutoContextCascade(
        SingleHeadStage(IN_CHANNELS, LSD_CHANNELS, "lsd"),
        SingleHeadStage(LSD_CHANNELS + IN_CHANNELS, AFF_CHANNELS, "aff"),
        include_raw=True,
    )
    out = model(torch.randn(*SHAPE))
    assert out["output"]["aff"].shape == (1, AFF_CHANNELS) + SHAPE[2:]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _cfg(params):
    return OmegaConf.create(
        {
            "model": {
                "arch": {"type": "rsunet_aclsd", "params": params},
                "in_channels": IN_CHANNELS,
                "out_channels": AFF_CHANNELS,
                "external_weights_path": None,
                "external_weights_key_prefix": "",
                "rsunet": {
                    "width": [8, 16],
                    "norm": "group",
                    "num_groups": 4,
                    "activation": "relu",
                    "down_factors": None,
                    "depth_2d": 0,
                    "kernel_2d": [1, 3, 3],
                    "act_negative_slope": 0.01,
                    "act_init": 0.25,
                },
            }
        }
    )


def test_builder_is_registered():
    assert is_architecture_available("rsunet_aclsd")
    assert get_architecture_builder("rsunet_aclsd") is build_rsunet_aclsd


def test_builder_constructs_working_acrlsd():
    cfg = _cfg({"lsd_channels": LSD_CHANNELS, "aff_channels": AFF_CHANNELS, "include_raw": True})
    model = build_rsunet_aclsd(cfg)

    assert isinstance(model, AutoContextCascade)
    assert model.include_raw is True
    assert model.stage2.input_conv.pre[1].in_channels == LSD_CHANNELS + IN_CHANNELS

    out = model(torch.randn(*SHAPE))
    assert out["output"]["lsd"].shape[1] == LSD_CHANNELS
    assert out["output"]["aff"].shape[1] == AFF_CHANNELS


def test_builder_constructs_aclsd_input_width():
    cfg = _cfg({"lsd_channels": LSD_CHANNELS, "include_raw": False})
    model = build_rsunet_aclsd(cfg)
    assert model.include_raw is False
    assert model.stage2.input_conv.pre[1].in_channels == LSD_CHANNELS


def test_builder_freeze_flag():
    cfg = _cfg({"freeze_stage1": True})
    model = build_rsunet_aclsd(cfg)
    assert model.freeze_stage1 is True
    assert all(not p.requires_grad for p in model.stage1.parameters())
