# Offset Metadata Preservation

When reading data from Neuroglancer precomputed format with ROI specifications, the pipeline automatically captures and preserves spatial offset information. This metadata can then be included when saving outputs as OME-Zarr format, maintaining the spatial coordinate system.

## How It Works

1. **Reading with ROI**: When you specify an ROI in a precomputed URL, the reader captures:
   - Offset (z,y,x) in voxels - the starting coordinates of the ROI
   - Resolution (z,y,x) in nanometers - the voxel size
   - MIP level - the pyramid level being read

2. **Metadata Storage**: This information is stored in a global metadata dictionary `_VOLUME_METADATA` using the source URL as the key.

3. **Writing with Offset**: When saving as OME-Zarr, the writer automatically retrieves and includes this metadata in the OME-NGFF format:
   - Translation transform: captures the spatial offset
   - Scale transform: captures the voxel resolution (converted to micrometers)

## Example Usage

### Basic ROI Read and Write

```python
from connectomics.data.utils import readvol, write_ome_zarr

# Read a region from precomputed volume
# ROI format: @mip#z_start-z_end_y_start-y_end_x_start-x_end
source_url = "precomputed://gs://my-bucket/em-data@1#0-100_0-200_300-500"
vol = readvol(source_url)

# The offset [0, 0, 300] and resolution are automatically captured

# Save as OME-Zarr with metadata
write_ome_zarr(
    "output.ome.zarr", 
    vol, 
    dataset='0', 
    multiscale=True,
    source=source_url  # Use metadata from this source
)
```

### Explicit Offset and Resolution

You can also explicitly provide offset and resolution if needed:

```python
# Read from any source
vol = readvol("input.h5", dataset='main')

# Save with custom spatial metadata
write_ome_zarr(
    "output.ome.zarr",
    vol,
    dataset='0',
    multiscale=True,
    offset=[1000, 2000, 3000],  # (z,y,x) in voxels
    resolution=[40.0, 4.0, 4.0]  # (z,y,x) in nanometers
)
```

### Inference Pipeline

For inference pipelines, the metadata is automatically used:

```bash
# Run inference on a ROI
python scripts/main.py \
    --config-file configs/my_model.yaml \
    --inference \
    DATASET.IMAGE_NAME precomputed://gs://bucket/data@2#100-200_1000-2000_1000-2000 \
    INFERENCE.OUTPUT_NAME output.ome.zarr
```

The output file will automatically include:
- Translation: [100, 1000, 1000] voxels (z,y,x offset)
- Scale: [resolution_z, resolution_y, resolution_x] micrometers

## OME-NGFF Metadata Structure

The offset and resolution are stored in the `coordinateTransformations` field:

```json
{
  "multiscales": [{
    "version": "0.4",
    "datasets": [{
      "path": "0",
      "coordinateTransformations": [
        {
          "type": "translation",
          "translation": [100, 1000, 1000]  // z,y,x offset in voxels
        },
        {
          "type": "scale",
          "scale": [0.04, 0.004, 0.004]  // z,y,x resolution in μm
        }
      ]
    }]
  }]
}
```

## Benefits

1. **Coordinate System Preservation**: When processing sub-regions, the global coordinates are maintained
2. **OME-NGFF Compliance**: Uses standard OME-NGFF coordinate transformation specification
3. **Automatic Handling**: No manual coordinate tracking needed
4. **Resolution Awareness**: Physical spacing is preserved for accurate measurements

## Notes

- Offsets are stored in (z,y,x) order following the pipeline convention
- Resolutions are converted from nanometers to micrometers in OME-NGFF metadata
- For 4D volumes (c,z,y,x), translation applies only to spatial dimensions
- If multiple volumes are read, the last metadata entry is used as fallback
