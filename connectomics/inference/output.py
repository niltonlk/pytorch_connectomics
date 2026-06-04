"""Output writing utilities for inference and tuning."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig

from ..config import Config
from ..data.processing.nnunet_preprocess import restore_prediction_to_input_space

logger = logging.getLogger(__name__)


def _resolve_output_formats(inference_cfg: Any) -> List[str]:
    """Normalize ``inference.save_backend`` to a non-empty list of backend names."""
    backend = getattr(inference_cfg, "save_backend", "h5")

    if isinstance(backend, (list, tuple)):
        formats = [str(item).strip() for item in backend if str(item).strip()]
        return formats or ["h5"]

    backend_str = str(backend).strip()
    if not backend_str:
        return ["h5"]

    if "," in backend_str:
        formats = [part.strip() for part in backend_str.split(",") if part.strip()]
        return formats or ["h5"]

    return [backend_str]


def _write_neuroglancer_precomputed(
    output_dir: Path,
    artifact_stem: str,
    sample: np.ndarray,
    *,
    resolution_xyz: tuple[float, float, float],
) -> Path:
    """Write one 3D/4D volume as single-scale Neuroglancer precomputed (raw encoding).

    The writer exports a single chunk and records voxel resolution in xyz order.
    Supported layouts:
    - ``(Z, Y, X)`` (single-channel)
    - ``(C, Z, Y, X)`` (channel-first multi-channel)
    """
    if sample.ndim not in (3, 4):
        raise ValueError(
            "Neuroglancer precomputed export supports (Z,Y,X) or (C,Z,Y,X) "
            f"(got shape {sample.shape})."
        )

    dtype_name = np.dtype(sample.dtype).name
    if dtype_name not in {
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    }:
        raise ValueError(
            "Unsupported dtype for Neuroglancer precomputed export: "
            f"{sample.dtype}."
        )

    if sample.ndim == 3:
        num_channels = 1
        z, y, x = (int(v) for v in sample.shape)
    else:
        num_channels = int(sample.shape[0])
        z, y, x = (int(v) for v in sample.shape[1:])
        if num_channels < 1:
            raise ValueError("Neuroglancer precomputed requires at least one channel.")

    out_dir = output_dir / f"{artifact_stem}.precomputed"

    try:
        from cloudvolume import CloudVolume
    except Exception as exc:
        raise ModuleNotFoundError(
            "CloudVolume is required for neuroglancer_precomputed export. "
            "Install cloud-volume>=11.0.0."
        ) from exc

    info = CloudVolume.create_new_info(
        num_channels=num_channels,
        layer_type="image",
        data_type=dtype_name,
        encoding="raw",
        resolution=list(resolution_xyz),
        voxel_offset=[0, 0, 0],
        chunk_size=[x, y, z],
        volume_size=[x, y, z],
    )
    vol = CloudVolume(
        f"file://{out_dir}",
        info=info,
        compress=False,
        progress=False,
        parallel=False,
        fill_missing=True,
    )
    vol.commit_info()
    vol.commit_provenance()

    # CloudVolume expects XYZC order for image layers.
    if sample.ndim == 3:
        data_xyzc = np.ascontiguousarray(sample.transpose(2, 1, 0)[..., np.newaxis])
    else:
        data_xyzc = np.ascontiguousarray(sample.transpose(3, 2, 1, 0))
    vol[:, :, :] = data_xyzc
    return out_dir


def _resolve_output_resolution_zyx(data_cfg: Any, mode: str) -> tuple[float, float, float]:
    """Resolve voxel resolution from merged data config as ZYX spacing."""
    if data_cfg is None:
        return (1.0, 1.0, 1.0)

    section = getattr(data_cfg, mode, None)
    resolution = getattr(section, "resolution", None) if section is not None else None
    if resolution is None:
        resolution = getattr(data_cfg, "resolution", None)

    if resolution is None:
        return (1.0, 1.0, 1.0)

    values = tuple(float(v) for v in resolution)
    if len(values) != 3:
        logger.warning(
            f"Expected 3D resolution for output metadata, got {resolution!r}; "
            "falling back to [1,1,1]."
        )
        return (1.0, 1.0, 1.0)
    return values


def _write_zarr_attrs(
    out_path: Path,
    sample: np.ndarray,
    *,
    resolution_zyx: tuple[float, float, float],
) -> None:
    """Attach standard metadata attrs to zarr prediction arrays."""
    try:
        import zarr

        arr = zarr.open(str(out_path), mode="a")
        num_channels = int(sample.shape[0]) if sample.ndim == 4 else 1
        arr.attrs["kind"] = "raw_prediction"
        arr.attrs["layout"] = "CZYX" if sample.ndim == 4 else "ZYX"
        arr.attrs["num_channels"] = num_channels
        arr.attrs["resolution_zyx"] = [
            float(resolution_zyx[0]),
            float(resolution_zyx[1]),
            float(resolution_zyx[2]),
        ]
        arr.attrs["resolution_xyz"] = [
            float(resolution_zyx[2]),
            float(resolution_zyx[1]),
            float(resolution_zyx[0]),
        ]
        arr.attrs["axes"] = "czyx" if sample.ndim == 4 else "zyx"
    except Exception as exc:
        logger.warning(f"Failed to attach zarr attrs for {out_path.name}: {exc}")


def resolve_output_filenames(
    cfg: Config | DictConfig, batch: Dict[str, Any], global_step: int = 0
) -> List[str]:
    """Extract and resolve per-volume directory names from batch metadata.

    Returns one stem per batch item, used as the per-volume subdir under
    ``<inference.save_path>``. When the raw filename stem is uninformative
    (e.g. ``img`` from ``data.zarr/img``), falls back to the parent dir name
    via the same rule as ``output_naming._stem_from_image_path``.
    """
    from connectomics.runtime.output_naming import _stem_from_image_path
    images = batch.get("image")
    if images is not None:
        if isinstance(images, (str, os.PathLike)):
            batch_size = 1
        elif isinstance(images, (list, tuple)):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]
    else:
        batch_size = 1

    meta = batch.get("image_meta_dict")
    filenames: List[str] = []

    if isinstance(meta, list):
        for meta_item in meta:
            if isinstance(meta_item, dict):
                filename = meta_item.get("filename_or_obj")
                if filename is not None:
                    filenames.append(filename)
        batch_size = max(batch_size, len(filenames))
    elif isinstance(meta, dict):
        meta_filenames = meta.get("filename_or_obj")
        if isinstance(meta_filenames, (list, tuple)):
            filenames = [f for f in meta_filenames if f is not None]
        elif meta_filenames is not None:
            filenames = [meta_filenames]
        if len(filenames) > 0:
            batch_size = max(batch_size, len(filenames))

    if not filenames:
        if isinstance(images, (str, os.PathLike)):
            filenames = [str(images)]
        elif isinstance(images, (list, tuple)):
            filenames = [str(image) for image in images if isinstance(image, (str, os.PathLike))]
            if filenames:
                batch_size = max(batch_size, len(filenames))

    resolved_names: List[str] = []
    for idx in range(batch_size):
        if idx < len(filenames) and filenames[idx]:
            resolved_names.append(_stem_from_image_path(filenames[idx]))
        else:
            resolved_names.append(f"volume_{global_step}_{idx}")

    if len(resolved_names) < batch_size:
        logger.warning(
            f"resolve_output_filenames - Only {len(resolved_names)} "
            f"filenames but batch_size is {batch_size}, padding with fallback names"
        )
        while len(resolved_names) < batch_size:
            resolved_names.append(f"volume_{global_step}_{len(resolved_names)}")

    return resolved_names


def _extract_meta_for_index(batch_meta: Any, idx: int) -> Dict[str, Any]:
    if batch_meta is None:
        return {}
    if isinstance(batch_meta, list):
        if idx < len(batch_meta) and isinstance(batch_meta[idx], dict):
            return dict(batch_meta[idx])
        return {}
    if isinstance(batch_meta, dict):
        out: Dict[str, Any] = {}
        for key, value in batch_meta.items():
            if isinstance(value, (list, tuple)) and idx < len(value):
                out[key] = value[idx]
            else:
                out[key] = value
        return out
    return {}


def _should_restore_outputs(cfg: Config | DictConfig, mode: str) -> bool:
    _inference_cfg, data_cfg, _output_dir = _resolve_mode_configs(cfg, mode)
    if data_cfg is not None:
        pre = getattr(data_cfg, "nnunet_preprocessing", None)
        if pre is not None:
            return bool(
                getattr(pre, "enabled", False) and getattr(pre, "restore_to_input_space", False)
            )

    return False


def _resolve_mode_configs(
    cfg: Config | DictConfig,
    mode: str,
) -> tuple[Any, Any, str | None]:
    """Resolve merged runtime inference/data/output configuration."""
    inference_cfg = getattr(cfg, "inference", None)
    output_dir_value = getattr(inference_cfg, "save_path", None) if inference_cfg is not None else None

    # Resolve data config: prefer stage-specific data (cfg.test.data / cfg.tune.data),
    # fall back to cfg.data (post-stage-resolution)
    data_cfg = None
    if mode == "test" and hasattr(cfg, "test") and hasattr(cfg.test, "data"):
        data_cfg = cfg.test.data
    elif mode == "tune" and hasattr(cfg, "tune") and hasattr(cfg.tune, "data"):
        data_cfg = cfg.tune.data
    if data_cfg is None:
        data_cfg = getattr(cfg, "data", None)

    # For decode-only mode: output to same folder as the input prediction
    if not output_dir_value:
        saved_pred = getattr(getattr(cfg, "decoding", None), "load_prediction_path", "")
        if saved_pred:
            from pathlib import Path

            output_dir_value = str(Path(saved_pred).expanduser().parent)

    return inference_cfg, data_cfg, output_dir_value


def _convert_intensity_dtype(
    data: np.ndarray,
    target_dtype_str: str | None,
    *,
    config_name: str,
) -> np.ndarray:
    """Convert prediction/output dtype using the common intensity dtype vocabulary."""
    if target_dtype_str is None:
        return data

    dtype_map = {
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }

    if target_dtype_str not in dtype_map:
        logger.warning(
            f"Unknown dtype '{target_dtype_str}' in {config_name}. "
            f"Supported: {list(dtype_map.keys())}. Keeping current dtype."
        )
        return data

    target_dtype = dtype_map[target_dtype_str]
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        data = np.clip(data, info.min, info.max)
        logger.info(
            f"Converting to {target_dtype_str} for {config_name} "
            f"(clipped to [{info.min}, {info.max}])"
        )
    else:
        logger.info(f"Converting to {target_dtype_str} for {config_name}")

    return data.astype(target_dtype, copy=False)


def _apply_intensity_transform(
    data: np.ndarray,
    *,
    intensity_scale: float | None,
    intensity_dtype: str | None,
    config_name: str,
) -> np.ndarray:
    """Apply intensity scaling and optional dtype conversion."""
    if intensity_scale is not None and intensity_scale >= 0:
        data = data.astype(np.float32, copy=False)
        if intensity_scale != 1.0:
            data = data * float(intensity_scale)
            logger.info(
                f"Scaled predictions by {intensity_scale} for {config_name} -> "
                f"range [{data.min():.4f}, {data.max():.4f}]"
            )
    else:
        logger.info(
            f"Intensity scaling disabled for {config_name} "
            f"(scale={intensity_scale} < 0), keeping raw predictions"
        )

    return _convert_intensity_dtype(data, intensity_dtype, config_name=config_name)


def apply_prediction_transform(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """Apply semantic prediction transforms before decoding/evaluation."""
    if not hasattr(cfg, "inference"):
        return data

    transform_cfg = getattr(cfg.inference, "prediction_transform", None)
    if transform_cfg is None or not getattr(transform_cfg, "enabled", False):
        return data

    return _apply_intensity_transform(
        data,
        intensity_scale=getattr(transform_cfg, "intensity_scale", -1.0),
        intensity_dtype=getattr(transform_cfg, "intensity_dtype", None),
        config_name="inference.prediction_transform",
    )


def apply_storage_dtype_transform(cfg: Config | DictConfig, data: np.ndarray) -> np.ndarray:
    """Apply save/cache-only dtype conversion. Skip when ``inference.save_dtype`` is unset."""
    if not hasattr(cfg, "inference"):
        return data

    storage_dtype = getattr(cfg.inference, "save_dtype", None)
    if storage_dtype is None:
        return data

    return _convert_intensity_dtype(
        data,
        storage_dtype,
        config_name="inference.save_dtype",
    )


def _strip_extension(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return Path(name).stem if "." in Path(name).name else name


def write_outputs(
    cfg: Config | DictConfig,
    predictions: np.ndarray,
    filenames: List[str],
    suffix: str = "prediction.h5",
    mode: str = "test",
    batch_meta: Any = None,
) -> None:
    """Persist predictions under ``<save_path>/<volume_stem>/<suffix>``.

    ``filenames`` are the per-volume directory names (typically the image
    file stems with parent-dir fallback for uninformative stems like
    ``img``). ``suffix`` is the artifact filename inside the per-volume
    subdir; an extension matching the configured backend is appended if
    missing.

    Note: output_transpose is NOT applied here. It is applied once in the
    decoding stage to avoid double-transpose when decoded data are then saved.
    """
    inference_cfg, data_cfg, output_dir_value = _resolve_mode_configs(cfg, mode)
    if inference_cfg is None:
        return

    if not output_dir_value:
        return

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    from connectomics.data.io import save_volume, write_hdf5

    if predictions.ndim >= 4:
        actual_batch_size = predictions.shape[0]
    elif predictions.ndim == 3:
        if len(filenames) > 0 and predictions.shape[0] == len(filenames):
            actual_batch_size = predictions.shape[0]
        else:
            actual_batch_size = 1
            predictions = predictions[np.newaxis, ...]
    else:
        actual_batch_size = 1
        predictions = predictions[np.newaxis, ...]

    if len(filenames) != actual_batch_size:
        logger.warning(
            f"write_outputs - filename count ({len(filenames)}) "
            f"does not match batch size ({actual_batch_size}). Using first "
            f"{min(len(filenames), actual_batch_size)} filenames."
        )

    should_restore = _should_restore_outputs(cfg, mode) and (
        suffix == "prediction.h5" or suffix == "prediction"
    )

    artifact_stem = _strip_extension(suffix)
    resolution_zyx = _resolve_output_resolution_zyx(data_cfg, mode)
    resolution_xyz = (resolution_zyx[2], resolution_zyx[1], resolution_zyx[0])

    for idx in range(actual_batch_size):
        if idx >= len(filenames):
            logger.warning(f"write_outputs - no filename for batch index {idx}, skipping")
            continue

        sample = predictions[idx]
        if should_restore:
            sample_meta = _extract_meta_for_index(batch_meta, idx)
            sample = restore_prediction_to_input_space(sample, sample_meta)

        volume_stem = filenames[idx]
        sample = np.squeeze(sample)

        output_formats = _resolve_output_formats(inference_cfg)

        sample = apply_storage_dtype_transform(cfg, sample)

        volume_dir = output_dir / volume_stem
        volume_dir.mkdir(parents=True, exist_ok=True)

        for fmt in output_formats:
            fmt_lower = fmt.lower()

            if fmt_lower == "h5":
                out_path = volume_dir / f"{artifact_stem}.h5"
                write_hdf5(out_path, sample, dataset="main")
                logger.info(f"Saved HDF5: {volume_stem}/{out_path.name}")

            elif fmt_lower in ["tif", "tiff"]:
                out_path = volume_dir / f"{artifact_stem}.tiff"
                try:
                    save_volume(str(out_path), sample, file_format="tiff")
                    logger.info(
                        f"Saved TIFF: {volume_stem}/{out_path.name} (shape: {sample.shape})"
                    )
                except Exception as exc:
                    logger.warning(f"TIFF export failed: {exc}")

            elif fmt_lower in ["nii", "nii.gz"]:
                out_path = volume_dir / f"{artifact_stem}.nii.gz"
                try:
                    save_volume(str(out_path), sample, file_format="nii.gz")
                    logger.info(f"Saved NIfTI: {volume_stem}/{out_path.name}")
                except Exception as exc:
                    logger.warning(f"NIfTI export failed: {exc}")

            elif fmt_lower == "zarr":
                out_path = volume_dir / f"{artifact_stem}.zarr"
                try:
                    save_volume(str(out_path), sample, file_format="zarr")
                    _write_zarr_attrs(out_path, sample, resolution_zyx=resolution_zyx)
                    logger.info(f"Saved Zarr: {volume_stem}/{out_path.name}")
                except Exception as exc:
                    logger.warning(f"Zarr export failed: {exc}")

            elif fmt_lower in [
                "neuroglancer_precomputed",
                "precomputed",
                "neuroglancer",
                "ng_precomputed",
            ]:
                try:
                    out_path = _write_neuroglancer_precomputed(
                        volume_dir,
                        artifact_stem,
                        sample,
                        resolution_xyz=resolution_xyz,
                    )
                    logger.info(f"Saved Neuroglancer precomputed: {volume_stem}/{out_path.name}/")
                except Exception as exc:
                    logger.warning(f"Neuroglancer precomputed export failed: {exc}")

            elif fmt_lower == "png":
                out_dir = volume_dir / f"{artifact_stem}_png"
                try:
                    save_volume(str(out_dir), sample, file_format="png")
                    logger.info(f"Saved PNG stack: {volume_stem}/{out_dir.name}/")
                except Exception as exc:
                    logger.warning(f"PNG export failed: {exc}")

            else:
                logger.warning(
                    "Unknown format '%s' - skipping. Supported: h5, zarr, tiff, "
                    "nii.gz, png, neuroglancer_precomputed"
                    % fmt
                )


__all__ = [
    "apply_prediction_transform",
    "apply_storage_dtype_transform",
    "resolve_output_filenames",
    "write_outputs",
]
