# Local Shape Descriptors (LSD) & Auto-Context in PyTorch Connectomics

Reference guide for the LSD target and the auto-context (ACLSD / ACRLSD)
cascades. Based on Sheridan et al., *Local shape descriptors for neuron
segmentation* (Nature Methods 2022) and the `funkelab/lsd` implementation.

---

## 1. What LSDs are

A Local Shape Descriptor is a per-voxel summary of the local segment
neighborhood (Gaussian-weighted by default). In **3D** it has **10 channels**:

| channels | meaning                              | normalized |
|----------|--------------------------------------|------------|
| `[0:3]`  | mean offset in (z, y, x)             | → `[0, 1]` |
| `[3:6]`  | variance along (z, y, x)             | → `[0, 1]` |
| `[6:9]`  | Pearson covariance (zy, zx, yx)      | → `[0, 1]` |
| `[9]`    | local segment size                   | → `[0, 1]` |

**2D** has the 6-channel analogue (`[0:2]` offset, `[2:4]` variance, `[4]`
Pearson, `[5]` size). Signed channels are shifted into `[0, 1]`; every channel
is **zero in background**.

Implementation: [`connectomics/data/processing/lsd.py`](../connectomics/data/processing/lsd.py)
(`seg_to_lsd`, `LsdExtractor`) — a numpy port of the reference with the
`gunpowder` dependency stripped and a bbox-optimized fast path. It is a
registered `lsd` target in the label-transform registry, so any config can emit
it.

---

## 2. MTLSD — single-pass multi-task (already supported)

LSDs are produced in the **data pipeline** (model-agnostic), and the loss
orchestrator routes supervision either by **named head** or by **channel slice**
on a single-tensor model. Two ready-made setups:

### 2a. MedNeXt multi-head
[`tutorials/neuron_nisb/base_banis+_lsd.yaml`](../tutorials/neuron_nisb/base_banis+_lsd.yaml) —
a 6-channel `aff` head and a 10-channel `lsd` head on a shared MedNeXt trunk,
supervised in parallel. Decoding consumes the `aff` head's short-range channels
`[0:3]`.

### 2b. Single-tensor models (RSUNet / MONAI)
Set `out_channels: 16` and route two channel-sliced loss terms:

```yaml
model:
  arch: { type: rsunet }
  out_channels: 16          # 6 affinity + 10 LSD
  loss:
    losses:
      - { function: PerChannelBCEWithLogitsLoss, pred_slice: "0:6",  target_slice: "0:6"  }
      - { function: WeightedBCEWithLogitsLoss,   pred_slice: "6:16", target_slice: "6:16" }
```

RSUNet has a single `output_head` (16 channels then sliced), so tasks share
everything up to the last conv — slightly more sharing than the paper's separate
final conv per task.

---

## 3. Auto-context (ACLSD / ACRLSD) — the cascade

The paper improves affinities by first predicting LSDs, then predicting
affinities **from the LSDs**. PyTC implements this as a **single end-to-end
two-stage cascade** (not the paper's offline two-run pipeline), so it fits the
existing Lightning loop with zero loss-system changes:

```
stage 1:  raw               -> LSD logits        (supervised: lsd head)
bridge:   sigmoid(LSD)      -> [0, 1] feed
stage 2:  [LSD_feed (+raw)] -> affinity logits   (supervised: aff head)
```

The wrapper returns the standard named-head contract

```python
{"output": {"lsd": lsd_logits, "aff": aff_logits}}
```

so the loss orchestrator routes `pred_head: lsd` and `pred_head: aff` unchanged.
The LSD tensor handed to the loss is **raw logits** (`*WithLogits`-friendly);
only the bridge into stage 2 is sigmoided (LSDs are bounded in `[0, 1]`, matching
the reference input).

### 3a. ACLSD vs ACRLSD
- **ACLSD** (`include_raw: false`) — stage 2 sees the LSD feed only
  (stage-2 input width = 10).
- **ACRLSD** (`include_raw: true`, default) — stage 2 sees the LSD feed
  concatenated with the raw image (stage-2 input width = 11).

### 3b. Gradient modes (`model.arch.params`)
| flag | behavior |
|------|----------|
| `detach_stage1: true` (default) | decouple the stages in one run (paper-like: affinity grads do not shape the LSDs). |
| `detach_stage1: false` | affinity gradients also flow into the LSDs. |
| `freeze_stage1: true` | literal paper setup: stage 1 frozen + kept in **eval** mode (implies detach). Combine with a pre-trained LSD net. |

Stage 1 always remains trainable from **its own** LSD loss unless frozen.

### 3c. Two backbones
- [`connectomics/models/architectures/mednext_autocontext.py`](../connectomics/models/architectures/mednext_autocontext.py)
  — `MedNeXtAutoContextWrapper`, registered as **`mednext_autocontext`**. Uses
  the MedNeXt `forward_features` / `heads` seam. Config:
  [`tutorials/neuron_nisb/base_banis+_aclsd.yaml`](../tutorials/neuron_nisb/base_banis+_aclsd.yaml).
- [`connectomics/models/architectures/autocontext.py`](../connectomics/models/architectures/autocontext.py)
  — the generic `AutoContextCascade`, registered as **`rsunet_aclsd`**. Treats
  each stage as a black box (`stage(x) -> tensor`, unwrapped by
  `_extract_main_tensor`), so it also works with MONAI or single-head MedNeXt.
  Config: [`tutorials/neuron_nisb/rsunet_aclsd.yaml`](../tutorials/neuron_nisb/rsunet_aclsd.yaml).

Example cascade config block:

```yaml
model:
  arch:
    type: rsunet_aclsd          # or mednext_autocontext
    params:
      lsd_channels: 10
      aff_channels: 6
      include_raw: true         # true -> ACRLSD, false -> ACLSD
      detach_stage1: true
      freeze_stage1: false
  in_channels: 1
  out_channels: 6               # = aff_channels
  primary_head: aff
```

---

## 4. Loss choice: BCE-with-logits vs sigmoid + MSE

The configs supervise LSDs with `WeightedBCEWithLogitsLoss` on raw logits rather
than the paper's sigmoid + MSE. Both are minimized at `p = t`, so the swap is
legitimate for LSD targets in `[0, 1]`. The difference is the gradient w.r.t. the
logit `z` (`p = sigmoid(z)`):

- BCE-with-logits: `dL/dz = p - t`.
- sigmoid + MSE: `dL/dz = 2(p - t)·p(1-p)` — the extra `p(1-p)` factor vanishes
  when the network is confidently wrong, so learning stalls.

BCE therefore trains faster/more stably from bad inits and is numerically fused
(log-sum-exp). Near the target both reduce to a linear restoring force, which is
why they behave equivalently in low-error regions. Note: the repo's
`WeightedMSELoss` has **no** sigmoid (`(logit - target)²`), so a naive MSE swap
is a regression — a faithful variant would need a small sigmoid-then-MSE loss.

---

## 5. Saving auxiliary heads at inference

With a multi-head / cascade model, `inference.model.head: aff` selects the head
that feeds decoding; `inference.save_all_heads: true` re-predicts each other head
(e.g. `lsd`) and writes it to its own file. Short-range vs long-range are
**channels within the `aff` head** (`[0:3]` short, `[3:6]` long), selected with
`inference.model.select_channel: [0, 1, 2]` — not separate heads.

---

## 6. Verifying the integration

```bash
python scripts/verify_lsd_integration.py          # dependency-light regression checks
python -m pytest tests/unit/test_autocontext_cascade.py \
                 tests/unit/test_mednext_autocontext.py -q
python scripts/validate_tutorial_configs.py \
    --glob 'tutorials/neuron_nisb/rsunet_aclsd.yaml' \
    --glob 'tutorials/neuron_nisb/base_banis+_aclsd.yaml'
```

The RSUNet cascade is exercised end-to-end (pure PyTorch); the MedNeXt cascade's
novel logic is tested against a stub trunk, since the real trunk needs the
optional `nnunet_mednext` dependency. Run the MedNeXt builder in an environment
with `nnunet_mednext` installed to exercise it for real.

---

## File inventory

**New**
- `connectomics/models/architectures/autocontext.py` — generic `AutoContextCascade` + `rsunet_aclsd`.
- `connectomics/models/architectures/mednext_autocontext.py` — `MedNeXtAutoContextWrapper` + `mednext_autocontext`.
- `tutorials/neuron_nisb/rsunet_aclsd.yaml`, `tutorials/neuron_nisb/base_banis+_aclsd.yaml`.
- `tests/unit/test_autocontext_cascade.py`, `tests/unit/test_mednext_autocontext.py`.
- `scripts/verify_lsd_integration.py`, `docs/LSD_INTEGRATION.md`.

**Edited**
- `connectomics/models/architectures/__init__.py` — register imports for both cascades.

**Pre-existing (not created here)**
- `connectomics/data/processing/lsd.py`, the `lsd` label target, and
  `tutorials/neuron_nisb/base_banis+_lsd.yaml`.
