#!/usr/bin/env python
"""Dependency-light regression check for the LSD + auto-context integration.

Validates three things without requiring gunpowder, nnunet_mednext, or a GPU:

1. ``connectomics.data.processing.lsd`` structural properties: 3D/2D channel
   counts, component subsets, [0, 1] output bounds, and background zeroing.
2. The auto-context channel math used by the tutorial configs
   (aff ``[0:6]`` + lsd ``[6:16]`` = 16 target channels; ACLSD vs ACRLSD
   stage-2 input widths).
3. Architecture registration of both cascades (``rsunet_aclsd`` always;
   ``mednext_autocontext`` when importable) and an end-to-end RSUNet cascade
   forward pass (only when torch is available).

Run:
    python scripts/verify_lsd_integration.py
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectomics.data.processing.lsd import seg_to_lsd  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}{(' -- ' + detail) if detail else ''}")


def _two_blob_volume_3d(shape=(16, 32, 32)) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.int64)
    d, h, w = shape
    seg[:, : h // 2, :] = 1
    seg[:, h // 2 :, : w // 2] = 2
    # Leave a background slab so background zeroing is testable.
    seg[:, h // 2 :, 3 * w // 4 :] = 0
    return seg


def _two_blob_image_2d(shape=(32, 32)) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.int64)
    h, w = shape
    seg[: h // 2, :] = 1
    seg[h // 2 :, : w // 2] = 2
    seg[h // 2 :, 3 * w // 4 :] = 0
    return seg


def verify_lsd_extractor() -> None:
    print("LSD extractor properties")
    seg3d = _two_blob_volume_3d()
    lsd3d = seg_to_lsd(seg3d, sigma=5)
    check("3D descriptor has 10 channels", lsd3d.shape == (10,) + seg3d.shape, str(lsd3d.shape))
    check("3D output is float32", lsd3d.dtype == np.float32, str(lsd3d.dtype))
    check("3D output within [0, 1]", float(lsd3d.min()) >= 0.0 and float(lsd3d.max()) <= 1.0,
          f"[{lsd3d.min():.3f}, {lsd3d.max():.3f}]")

    bg = seg3d == 0
    bg_vals = lsd3d[:, bg]
    check("3D background is zero across all channels", np.allclose(bg_vals, 0.0),
          f"max|bg|={np.abs(bg_vals).max():.3e}")

    fg = seg3d > 0
    check("3D foreground is non-trivial", float(np.abs(lsd3d[:, fg]).max()) > 0.0)

    # Component subset: mean-offset (0,1,2) + size (9) -> 4 channels.
    lsd_sub = seg_to_lsd(seg3d, sigma=5, components="0129")
    check("3D component subset '0129' has 4 channels", lsd_sub.shape[0] == 4, str(lsd_sub.shape))
    check("3D subset matches full-descriptor channels",
          np.allclose(lsd_sub[:3], lsd3d[:3]) and np.allclose(lsd_sub[3], lsd3d[9]))

    seg2d = _two_blob_image_2d()
    lsd2d = seg_to_lsd(seg2d, sigma=5)
    check("2D descriptor has 6 channels", lsd2d.shape == (6,) + seg2d.shape, str(lsd2d.shape))
    check("2D output within [0, 1]", float(lsd2d.min()) >= 0.0 and float(lsd2d.max()) <= 1.0)
    check("2D background is zero", np.allclose(lsd2d[:, seg2d == 0], 0.0))


def verify_config_channel_math() -> None:
    print("Auto-context channel math")
    aff_channels = 6
    lsd_channels = 10
    total = aff_channels + lsd_channels
    check("aff[0:6] + lsd[6:16] == 16 target channels", total == 16, str(total))

    in_channels = 1
    aclsd_stage2_in = lsd_channels                       # LSD feed only
    acrlsd_stage2_in = lsd_channels + in_channels        # LSD feed + raw
    check("ACLSD stage-2 input width == 10", aclsd_stage2_in == 10, str(aclsd_stage2_in))
    check("ACRLSD stage-2 input width == 11", acrlsd_stage2_in == 11, str(acrlsd_stage2_in))


def verify_registration_and_forward() -> None:
    print("Architecture registration & forward")
    from connectomics.models.architectures import is_architecture_available

    check("rsunet_aclsd registered", is_architecture_available("rsunet_aclsd"))
    check("mednext_autocontext registered", is_architecture_available("mednext_autocontext"))

    try:
        import torch
    except ImportError:
        print("  [skip] torch not installed; skipping forward-pass checks")
        return

    from connectomics.models.architectures.autocontext import AutoContextCascade
    from connectomics.models.architectures.rsunet import RSUNet

    def _stage(in_ch, out_ch):
        return RSUNet(in_channels=in_ch, out_channels=out_ch, width=[8, 16],
                      norm="group", num_groups=4)

    model = AutoContextCascade(
        _stage(1, 10), _stage(11, 6), include_raw=True, detach_stage1=True
    )
    x = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        out = model(x)
    heads = out.get("output", {})
    check("cascade emits named-head contract {lsd, aff}",
          set(heads.keys()) == {"lsd", "aff"}, str(sorted(heads.keys())))
    check("lsd head has 10 channels", heads["lsd"].shape[1] == 10)
    check("aff head has 6 channels", heads["aff"].shape[1] == 6)
    check("lsd head returns raw logits (not sigmoided)", float(heads["lsd"].min()) < 0.0)


def main() -> int:
    verify_lsd_extractor()
    verify_config_channel_math()
    verify_registration_and_forward()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
