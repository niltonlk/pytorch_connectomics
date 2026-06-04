from __future__ import annotations

import json

import numpy as np

from connectomics.config import Config
from connectomics.data.io import read_volume
from connectomics.inference.output import write_outputs
from connectomics.config.schema import TestConfig


def test_write_outputs_supports_zarr_backend(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path)
    cfg.inference.save_backend = "zarr"

    pred = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 1, 2, 3, 4)
    write_outputs(
        cfg=cfg,
        predictions=pred,
        filenames=["sample"],
        suffix="prediction",
        mode="test",
        batch_meta=None,
    )

    out_path = tmp_path / "sample" / "prediction.zarr"
    assert out_path.is_dir()

    restored = read_volume(str(out_path))
    np.testing.assert_array_equal(restored, pred[0, 0])


def test_write_outputs_supports_neuroglancer_precomputed_backend(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path)
    cfg.inference.save_backend = "neuroglancer_precomputed"

    pred = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 3, 4)
    write_outputs(
        cfg=cfg,
        predictions=pred,
        filenames=["sample"],
        suffix="prediction",
        mode="test",
        batch_meta=None,
    )

    out_dir = tmp_path / "sample" / "prediction.precomputed"
    info_path = out_dir / "info"

    assert info_path.is_file()

    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["type"] == "image"
    assert info["num_channels"] == 1
    assert info["data_type"] == "uint16"
    assert info["scales"][0]["size"] == [4, 3, 2]

    scale_key = info["scales"][0]["key"]
    chunk_path = out_dir / scale_key / "0-4_0-3_0-2"
    assert chunk_path.is_file()

    expected_nbytes = int(np.prod(pred.shape[-3:])) * np.dtype(np.uint16).itemsize
    assert chunk_path.stat().st_size == expected_nbytes


def test_write_outputs_supports_three_channel_neuroglancer_precomputed(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path)
    cfg.inference.save_backend = "neuroglancer_precomputed"

    pred = np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(1, 3, 2, 3, 4)
    write_outputs(
        cfg=cfg,
        predictions=pred,
        filenames=["sample"],
        suffix="prediction",
        mode="test",
        batch_meta=None,
    )

    out_dir = tmp_path / "sample" / "prediction.precomputed"
    info_path = out_dir / "info"

    assert info_path.is_file()

    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["type"] == "image"
    assert info["num_channels"] == 3
    assert info["data_type"] == "float32"
    assert info["scales"][0]["size"] == [4, 3, 2]

    scale_key = info["scales"][0]["key"]
    chunk_path = out_dir / scale_key / "0-4_0-3_0-2"
    assert chunk_path.is_file()

    expected_nbytes = int(np.prod(pred.shape[-3:])) * 3 * np.dtype(np.float32).itemsize
    assert chunk_path.stat().st_size == expected_nbytes


def test_write_outputs_supports_multiple_backends_in_one_run(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path)
    cfg.inference.save_backend = "h5,zarr"

    pred = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    write_outputs(
        cfg=cfg,
        predictions=pred,
        filenames=["sample"],
        suffix="prediction",
        mode="test",
        batch_meta=None,
    )

    assert (tmp_path / "sample" / "prediction.h5").is_file()
    assert (tmp_path / "sample" / "prediction.zarr").is_dir()


def test_write_outputs_writes_zarr_metadata_attrs(tmp_path):
    cfg = Config()
    cfg.inference.save_path = str(tmp_path)
    cfg.inference.save_backend = "zarr"

    pred = np.ones((1, 3, 2, 2, 2), dtype=np.float32)
    write_outputs(
        cfg=cfg,
        predictions=pred,
        filenames=["sample"],
        suffix="prediction",
        mode="test",
        batch_meta=None,
    )

    import zarr

    arr = zarr.open(str(tmp_path / "sample" / "prediction.zarr"), mode="r")
    assert arr.attrs["kind"] == "raw_prediction"
    assert arr.attrs["layout"] == "CZYX"
    assert arr.attrs["axes"] == "czyx"
    assert arr.attrs["num_channels"] == 3
    assert arr.attrs["resolution_zyx"] == [1.0, 1.0, 1.0]
    assert arr.attrs["resolution_xyz"] == [1.0, 1.0, 1.0]


def test_precomputed_info_uses_data_resolution_xyz(tmp_path):
    cfg = Config()
    cfg.test = TestConfig()
    cfg.inference.save_path = str(tmp_path)
    cfg.inference.save_backend = "neuroglancer_precomputed"
    cfg.test.data.test.resolution = [30, 6, 6]  # zyx

    pred = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 3, 4)
    write_outputs(
        cfg=cfg,
        predictions=pred,
        filenames=["sample"],
        suffix="prediction",
        mode="test",
        batch_meta=None,
    )

    info_path = tmp_path / "sample" / "prediction.precomputed" / "info"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["scales"][0]["resolution"] == [6, 6, 30]
