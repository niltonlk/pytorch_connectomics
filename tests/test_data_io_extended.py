import os
import tempfile
import importlib.util
import numpy as np
import pytest


def _load_data_io():
    """Load data_io module without importing the whole package tree."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mod_path = os.path.join(root, 'connectomics', 'data', 'utils', 'data_io.py')
    spec = importlib.util.spec_from_file_location('data_io', mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [
    (1, 4, 32, 32),   # (c,z,y,x) with spatial dims > 16
    (4, 32, 32),      # (z,y,x)
])
def test_read_omezarr_basic(shape):
    zarr = pytest.importorskip("zarr")
    dio = _load_data_io()

    # Create a temporary OME-Zarr store
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sample.ome.zarr")
        grp = zarr.open(path, mode='w')
        grp.create_dataset('0', data=np.arange(np.prod(shape), dtype=np.uint16).reshape(shape), chunks=True)
        # minimal multiscales-like attribute to point to '0'
        grp.attrs['multiscales'] = [{
            'version': '0.4',
            'datasets': [{'path': '0'}],
        }]

        vol = dio.readvol(path)
        assert vol.ndim in (3, 4)
        # channel-first if 4D
        if vol.ndim == 4:
            assert vol.shape[0] == 1


def test_read_precomputed_requires_roi_or_dep():
    # Should raise either ImportError (no cloud-volume) or ValueError (no ROI)
    dio = _load_data_io()
    src = 'precomputed://test-bucket/dataset'
    try:
        _ = dio.readvol(src)
    except (ImportError, ValueError):
        assert True
    else:
        # If neither error raised, reader attempted to fetch; this is unexpected in unit test
        pytest.skip("CloudVolume available and data source resolved; skipping in CI")


@pytest.mark.parametrize("shape,multiscale", [
    ((32, 64, 64), False),      # 3D (z,y,x)
    ((2, 32, 64, 64), False),   # 4D (c,z,y,x)
    ((32, 64, 64), True),       # 3D with multiscales metadata
])
def test_write_omezarr_basic(shape, multiscale):
    zarr = pytest.importorskip("zarr")
    dio = _load_data_io()

    # Create test data
    vol = np.random.randint(0, 255, shape, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "output.ome.zarr")
        
        # Write using write_ome_zarr
        dio.write_ome_zarr(path, vol, dataset='0', multiscale=multiscale)
        
        # Verify the file was created
        assert os.path.exists(path)
        
        # Read back and verify
        store = zarr.open(path, mode='r')
        assert '0' in store
        read_vol = np.array(store['0'])
        np.testing.assert_array_equal(read_vol, vol)
        
        # Check multiscales metadata if requested
        if multiscale:
            assert 'multiscales' in store.attrs
            assert len(store.attrs['multiscales']) > 0
