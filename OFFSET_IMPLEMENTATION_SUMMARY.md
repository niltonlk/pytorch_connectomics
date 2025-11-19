# Offset Metadata Preservation - Implementation Summary

## Feature Overview

This feature automatically captures and preserves spatial offset and resolution metadata when reading from Neuroglancer precomputed format and saves it to OME-Zarr format using the OME-NGFF coordinate transformation specification.

## Implementation Details

### 1. Global Metadata Storage

Added `_VOLUME_METADATA` dictionary in `data_io.py`:
```python
_VOLUME_METADATA = {}  # Global metadata storage for volume spatial information
```

Key structure:
```python
{
    'source_url': {
        'offset': [z, y, x],      # Start coordinates in voxels (z,y,x order)
        'resolution': [z, y, x],   # Voxel size in nm (z,y,x order)
        'mip': 1,                  # Pyramid level
        'url': 'base_url'          # Base URL without ROI
    }
}
```

### 2. Metadata Capture (readvol_precomputed)

When reading a precomputed volume with ROI:
```python
# Parse ROI from URL: precomputed://bucket/data@1#0-100_0-200_300-500
roi_slices = _parse_roi(roi_spec)  # Returns (xsl, ysl, zsl) slices
xsl, ysl, zsl = roi_slices

# Read from CloudVolume
cv = CloudVolume(url, mip=mip, ...)
vol = cv[xsl, ysl, zsl]

# Capture metadata (convert to z,y,x order)
offset_zyx = [zsl.start, ysl.start, xsl.start]
resolution = cv.resolution  # (x,y,z) in nm
resolution_zyx = [resolution[2], resolution[1], resolution[0]]

_VOLUME_METADATA[source] = {
    'offset': offset_zyx,
    'resolution': resolution_zyx,
    'mip': mip,
    'url': url,
}
```

### 3. Metadata Usage (write_ome_zarr)

Updated function signature:
```python
def write_ome_zarr(filename, vol, dataset=None, chunks=None, compression='blosc',
                   multiscale=False, offset=None, resolution=None, source=None)
```

Metadata retrieval logic:
1. If `offset`/`resolution` explicitly provided → use them
2. Else if `source` provided → lookup in `_VOLUME_METADATA[source]`
3. Else → fallback to last entry in `_VOLUME_METADATA` (for single-volume inference)

### 4. OME-NGFF Coordinate Transformations

Metadata is stored in the `coordinateTransformations` field:

```json
{
  "multiscales": [{
    "version": "0.4",
    "datasets": [{
      "path": "0",
      "coordinateTransformations": [
        {
          "type": "translation",
          "translation": [100.0, 1000.0, 500.0]  // z,y,x offset in voxels
        },
        {
          "type": "scale",
          "scale": [0.04, 0.004, 0.004]  // z,y,x resolution in μm (converted from nm)
        }
      ]
    }]
  }]
}
```

For 4D volumes (c,z,y,x):
```json
{
  "translation": [0.0, 100.0, 1000.0, 500.0],  // (c,z,y,x) - no offset for channel
  "scale": [1.0, 0.04, 0.004, 0.004]           // (c,z,y,x) - unit scale for channel
}
```

## Usage Examples

### 1. Automatic Metadata Preservation
```python
from connectomics.data.utils import readvol, write_ome_zarr

# Read with ROI - metadata automatically captured
vol = readvol("precomputed://gs://bucket/data@1#100-200_1000-2000_500-1000")

# Save with metadata - automatically retrieved
write_ome_zarr("output.ome.zarr", vol, dataset='0', multiscale=True,
               source="precomputed://gs://bucket/data@1#100-200_1000-2000_500-1000")
```

### 2. Explicit Metadata
```python
# Provide custom offset and resolution
write_ome_zarr("output.ome.zarr", vol, dataset='0', multiscale=True,
               offset=[500, 1000, 1500],        # (z,y,x) voxels
               resolution=[30.0, 5.0, 5.0])     # (z,y,x) nm
```

### 3. Inference Pipeline
```bash
# Metadata automatically flows through
python scripts/main.py --inference \
    DATASET.IMAGE_NAME "precomputed://gs://bucket/data@2#100-200_1000-2000_1000-2000" \
    INFERENCE.OUTPUT_NAME "output.ome.zarr"
```

## Test Coverage

Added 2 new tests to `test_data_io_extended.py`:

1. **test_write_omezarr_with_offset**: Tests explicit offset/resolution parameters
   - Verifies translation transform in metadata
   - Verifies scale transform (nm to μm conversion)
   - Checks coordinate transformation structure

2. **test_offset_metadata_from_precomputed**: Tests metadata flow from global dict
   - Simulates metadata storage as reader would
   - Verifies automatic retrieval via `source` parameter
   - Validates metadata application to output

Current test status: **7 passed, 1 skipped** (CloudVolume optional dependency)

## Files Modified

1. **connectomics/data/utils/data_io.py**:
   - Added `_VOLUME_METADATA` global dict
   - Modified `readvol_precomputed` to capture metadata
   - Updated `write_ome_zarr` signature and implementation
   - Added coordinate transformation generation

2. **tests/test_data_io_extended.py**:
   - Added `test_write_omezarr_with_offset`
   - Added `test_offset_metadata_from_precomputed`

3. **Documentation**:
   - Created `OFFSET_METADATA_EXAMPLE.md` with usage guide
   - Created `demo_offset_metadata.py` demonstration script

## Technical Notes

### Axis Order Convention
- **Pipeline**: (z,y,x) - depth-first, C-order
- **CloudVolume**: (x,y,z) - column-first, Fortran-order
- **OME-NGFF**: (z,y,x) for 3D, (c,z,y,x) for 4D

All conversions maintain this convention consistently.

### Resolution Units
- **Input**: Nanometers (nm) from CloudVolume
- **Output**: Micrometers (μm) for OME-NGFF compliance
- **Conversion**: `scale = resolution_nm / 1000.0`

### Metadata Lifetime
- Stored in global `_VOLUME_METADATA` dict
- Persists across multiple reads in same session
- Can be manually cleared: `data_io._VOLUME_METADATA.clear()`

## Benefits

1. **Coordinate System Preservation**: Sub-region processing maintains global coordinates
2. **OME-NGFF Compliance**: Uses standard OME-NGFF 0.4 coordinate transformations
3. **Automatic Handling**: No manual coordinate tracking needed in inference pipelines
4. **Resolution Awareness**: Physical spacing preserved for accurate measurements
5. **Flexible Interface**: Supports both automatic and explicit metadata specification

## Future Enhancements

Potential improvements for future versions:

1. **Multi-volume tracking**: Per-volume metadata keys instead of relying on source URL
2. **Metadata API**: Explicit functions to query/set metadata
3. **Rotation support**: Additional transformation types (rotation, shear)
4. **Metadata persistence**: Save/load metadata between sessions
5. **Validation**: Verify coordinate transformation consistency
