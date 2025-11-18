# Extended Volume Readers: OME-Zarr & Neuroglancer Precomputed

This feature branch adds first-class support for reading OME-Zarr and Neuroglancer "precomputed" volumes directly in the PyTorch Connectomics data loader. Both formats are optional (soft dependencies) and automatically detected by path/URL.

---

## 1. OME-Zarr Support

**Filename conventions:**
- Paths ending in `.zarr` or `.ome.zarr` are routed to the OME-Zarr reader.

**Optional dependencies:**
- Install `zarr` (and optionally `ome-zarr`) before use:
  ```bash
  pip install zarr
  # or for OME-NGFF metadata helpers:
  pip install ome-zarr
  ```

**Usage:**

A. **Auto-select multiscales dataset (default)**
```python
from connectomics.data.utils.data_io import readvol

vol = readvol('/path/to/data.ome.zarr')
# Picks first dataset in multiscales (usually '0')
```

B. **Select a specific array key**
```python
vol = readvol('/path/to/data.ome.zarr', dataset='0')
# or pass key in URL anchor:
vol = readvol('/path/to/data.ome.zarr#2')
```

**Axis and channel handling:**
- The reader automatically reorders 4D/5D arrays to match (c,z,y,x) or (z,y,x).
- Single-channel volumes can be auto-squeezed with `drop_channel=True`.

**Configuration in YACS:**
```yaml
DATASET:
  IMAGE_NAME: 'data.ome.zarr'
  # If needed, pass the dataset key via a '#' suffix or the dataset field
```

---

## 2. Neuroglancer Precomputed Support

**URL schemes:**
- `precomputed://`, `gs://`, `s3://`, or `file://` trigger the CloudVolume reader.

**Optional dependencies:**
- Install `cloud-volume`:
  ```bash
  pip install cloud-volume
  ```

**ROI requirement (important):**
- To avoid accidental massive downloads, you **must** provide an ROI in voxel coordinates.
- CloudVolume uses **x,y,z ordering** (Fortran-style), so specify: `x0:x1,y0:y1,z0:z1`
  - Either via the `dataset` argument: `readvol(url, dataset='x0:x1,y0:y1,z0:z1')`
  - Or appended as a URL anchor: `readvol('precomputed://bucket/seg#0:128,0:128,0:64')`

**Usage:**

A. **Public GCS source with ROI anchor (x,y,z order)**
```python
from connectomics.data.utils.data_io import readvol

# ROI: x=[1000,1064), y=[2000,2128), z=[3000,3064)
url = 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation#1000:1064,2000:2128,3000:3064'
vol = readvol(url)
# Returns shape (c,64,128,64) or (64,128,64) after transpose to z,y,x
```

B. **Local precomputed store**
```python
# x,y,z spans: [0,100) each
vol = readvol('file:///data/precomputed/myseg#0:100,0:100,0:100')
```

**Configuration in YACS:**
```yaml
DATASET:
  # ROI in x,y,z order: x=[500,564), y=[1000,1128), z=[2000,2128)
  IMAGE_NAME: 'precomputed://gs://bucket/dataset#500:564,1000:1128,2000:2128'
```

---

## 3. Fallback to Existing Formats

If the path does not match OME-Zarr or precomputed patterns, the existing readers for HDF5, TIFF, and PNG remain unchanged. This ensures backward compatibility.

---

## 4. Testing

Run the extended-format tests with:

```bash
pytest tests/test_data_io_extended.py -v
```

Requirements for tests:
- `zarr` for OME-Zarr test
- No cloud-volume needed (precomputed test is a smoke check that raises ImportError or ValueError)

---

## 5. Example Training Config

```yaml
DATASET:
  INPUT_PATH: '/data'
  IMAGE_NAME: 'sample.ome.zarr'
  LABEL_NAME: 'labels.ome.zarr'
  # Or use precomputed with ROI:
  # IMAGE_NAME: 'precomputed://gs://mybucket/images#0:512,0:512,0:512'
```

---

## 6. Known Limitations & Future Work

- **CloudVolume caching:** Currently disabled (`cache=False`) to avoid stale data. Future versions may integrate with a persistent cache.
- **Multiscale pyramids:** OME-Zarr reader picks the first dataset by default; explicit selection is supported via the `dataset` parameter.
- **Large ROIs:** For very large bounding boxes, consider loading via the existing tile/chunked dataset or converting to HDF5.
- **Axis ambiguity:** For unusual array shapes (e.g., all dimensions < 10), manual axis reordering may be needed.

---

## Contact & Contributions

Feedback, bug reports, and PRs are welcome! For questions, open an issue on the [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics) repository.
