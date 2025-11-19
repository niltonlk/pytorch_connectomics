# MIP Level Selection Examples

This guide shows how to select specific pyramid levels (MIP levels) when reading OME-Zarr and Neuroglancer precomputed data.

## Understanding MIP Levels

Multi-resolution pyramids store the same data at different resolutions:
- **MIP 0**: Full resolution (original data)
- **MIP 1**: 2x downsampled (half resolution)
- **MIP 2**: 4x downsampled (quarter resolution)
- **MIP 3**: 8x downsampled
- etc.

## OME-Zarr MIP Selection

OME-Zarr stores pyramid levels as separate arrays named '0', '1', '2', etc.

### Method 1: Using the `dataset` parameter

```python
from connectomics.data.utils.data_io import readvol

# Full resolution (MIP 0)
vol_full = readvol('/path/to/data.ome.zarr', dataset='0')

# MIP level 1 (2x downsampled)
vol_mip1 = readvol('/path/to/data.ome.zarr', dataset='1')

# MIP level 2 (4x downsampled)
vol_mip2 = readvol('/path/to/data.ome.zarr', dataset='2')
```

### Method 2: Using URL anchor syntax

```python
# MIP level 2
vol = readvol('/path/to/data.ome.zarr#2')
```

### In YACS configuration

```yaml
DATASET:
  INPUT_PATH: '/data'
  # Use MIP level 1 for faster training on downsampled data
  IMAGE_NAME: 'mydata.ome.zarr#1'
  LABEL_NAME: 'labels.ome.zarr#1'
```

## Neuroglancer Precomputed MIP Selection

Precomputed format uses the `@N` syntax in URLs to specify MIP level.

### URL syntax: `precomputed://path@MIP#ROI`

```python
from connectomics.data.utils.data_io import readvol

# Full resolution (MIP 0)
# ROI in x,y,z order: x=[1000,1064), y=[2000,2128), z=[3000,3064)
url = 'precomputed://gs://bucket/data#1000:1064,2000:2128,3000:3064'
vol_full = readvol(url)

# MIP level 1 (2x downsampled)
# Note: ROI coordinates should be in MIP 1 space (divided by 2)
url = 'precomputed://gs://bucket/data@1#500:532,1000:1064,1500:1532'
vol_mip1 = readvol(url)

# MIP level 2 (4x downsampled)
# ROI coordinates in MIP 2 space (divided by 4)
url = 'precomputed://gs://bucket/data@2#250:266,500:532,750:766'
vol_mip2 = readvol(url)
```

### In YACS configuration

```yaml
DATASET:
  INPUT_PATH: '/data'
  # Use MIP 1 for faster training
  IMAGE_NAME: 'precomputed://gs://mybucket/images@1#0:512,0:512,0:256'
  LABEL_NAME: 'precomputed://gs://mybucket/labels@1#0:512,0:512,0:256'
```

## Coordinate Space Considerations

**Important:** When using downsampled MIP levels, ROI coordinates must be in the coordinate space of that MIP level.

### Example: Same physical region at different MIPs

Let's say you want to read physical region x=[1000,1128), y=[2000,2256), z=[3000,3128):

```python
# MIP 0 (full resolution)
url_mip0 = 'precomputed://gs://bucket/data#1000:1128,2000:2256,3000:3128'

# MIP 1 (2x downsampled) - divide coordinates by 2
url_mip1 = 'precomputed://gs://bucket/data@1#500:564,1000:1128,1500:1564'

# MIP 2 (4x downsampled) - divide coordinates by 4
url_mip2 = 'precomputed://gs://bucket/data@2#250:282,500:564,750:782'

# All three will read the SAME physical region, just at different resolutions
vol_mip0 = readvol(url_mip0)  # shape: (128, 256, 128) in z,y,x
vol_mip1 = readvol(url_mip1)  # shape: (64, 128, 64)
vol_mip2 = readvol(url_mip2)  # shape: (32, 64, 32)
```

## Use Cases

### 1. Fast prototyping and debugging
```python
# Use MIP 2 for quick iteration
vol = readvol('/data/large_volume.ome.zarr', dataset='2')
```

### 2. Memory-efficient inference on large volumes
```python
# Process at MIP 1 to reduce memory usage
url = 'precomputed://gs://bucket/brain@1#0:1024,0:1024,0:512'
vol = readvol(url)
```

### 3. Multi-scale training
```python
# Train on different resolutions
vol_high = readvol('data.ome.zarr', dataset='0')
vol_low = readvol('data.ome.zarr', dataset='1')
```

## Verification

To verify which MIP level you're reading, check the output shape:

```python
vol = readvol('/path/to/data.ome.zarr', dataset='1')
print(f"Shape: {vol.shape}")
# For MIP 1, expect shape to be ~half of MIP 0 in each spatial dimension
```

## Summary Table

| Format | Syntax | Example | Notes |
|--------|--------|---------|-------|
| **OME-Zarr** | `path#N` or `dataset='N'` | `data.ome.zarr#1` | N is the array name ('0', '1', '2', etc.) |
| **Precomputed** | `url@N#roi` | `precomputed://gs://...@1#roi` | N is the MIP level (0, 1, 2, etc.) |

**Remember:** 
- MIP 0 = full resolution
- MIP N = 2^N times downsampled
- ROI coordinates must match the MIP level's coordinate space
