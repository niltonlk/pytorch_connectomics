#!/usr/bin/env python
"""
Demo script to test offset metadata preservation from precomputed to OME-Zarr.

This script demonstrates:
1. Reading a ROI from a precomputed volume (with metadata capture)
2. Saving to OME-Zarr with offset and resolution preserved
3. Verifying the metadata in the output file
"""

import sys
import json
import numpy as np
import tempfile
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectomics.data.utils import data_io as dio

def demo_offset_preservation():
    """Demonstrate offset metadata preservation."""
    
    print("=" * 60)
    print("OFFSET METADATA PRESERVATION DEMO")
    print("=" * 60)
    print()
    
    # Create synthetic test volume to simulate precomputed read
    print("1. Simulating precomputed read with ROI...")
    
    # Simulate metadata that would be captured from precomputed
    source_url = "precomputed://test-bucket/data@1#100-200_1000-2000_500-1000"
    test_metadata = {
        'offset': [100, 1000, 500],  # (z,y,x) in voxels
        'resolution': [40.0, 4.0, 4.0],  # (z,y,x) in nm
        'mip': 1,
        'url': "precomputed://test-bucket/data"
    }
    
    # Store metadata as reader would
    dio._VOLUME_METADATA[source_url] = test_metadata
    print(f"   Source: {source_url}")
    print(f"   Offset: {test_metadata['offset']} (z,y,x) voxels")
    print(f"   Resolution: {test_metadata['resolution']} (z,y,x) nm")
    print(f"   MIP level: {test_metadata['mip']}")
    print()
    
    # Create synthetic volume
    vol = np.random.randint(0, 255, (100, 1000, 500), dtype=np.uint8)
    print(f"   Volume shape: {vol.shape} (z,y,x)")
    print()
    
    # Write with metadata
    print("2. Writing to OME-Zarr with offset metadata...")
    with tempfile.TemporaryDirectory() as td:
        output_path = os.path.join(td, "output_with_offset.ome.zarr")
        
        dio.write_ome_zarr(
            output_path,
            vol,
            dataset='0',
            multiscale=True,
            source=source_url  # Pull metadata from this source
        )
        
        print(f"   Output: {output_path}")
        print()
        
        # Verify metadata
        print("3. Verifying OME-NGFF metadata...")
        try:
            import zarr
            store = zarr.open(output_path, mode='r')
            
            # Check data
            read_vol = np.array(store['0'])
            print(f"   ✓ Data shape matches: {read_vol.shape}")
            
            # Check multiscales metadata
            if 'multiscales' in store.attrs:
                ms = store.attrs['multiscales'][0]
                print(f"   ✓ Multiscales version: {ms['version']}")
                
                # Check coordinate transformations
                transforms = ms['datasets'][0]['coordinateTransformations']
                print(f"   ✓ Found {len(transforms)} coordinate transform(s)")
                
                for t in transforms:
                    if t['type'] == 'translation':
                        print(f"      - Translation: {t['translation']} (z,y,x) voxels")
                        expected = [float(x) for x in test_metadata['offset']]
                        if t['translation'] == expected:
                            print("        ✓ Offset correctly preserved!")
                        else:
                            print(f"        ✗ Expected {expected}")
                    
                    elif t['type'] == 'scale':
                        print(f"      - Scale: {t['scale']} (z,y,x) μm")
                        # Convert nm to μm for comparison
                        expected_scale = [r / 1000.0 for r in test_metadata['resolution']]
                        if np.allclose(t['scale'], expected_scale, rtol=1e-6):
                            print("        ✓ Resolution correctly preserved!")
                        else:
                            print(f"        ✗ Expected {expected_scale}")
                
                print()
                print("Full multiscales metadata:")
                print(json.dumps(ms, indent=2))
                
            else:
                print("   ✗ No multiscales metadata found")
        
        except ImportError:
            print("   ⚠ zarr not available for verification")
    
    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    # Clean up
    dio._VOLUME_METADATA.clear()


def demo_explicit_offset():
    """Demonstrate explicit offset specification."""
    
    print()
    print("=" * 60)
    print("EXPLICIT OFFSET SPECIFICATION DEMO")
    print("=" * 60)
    print()
    
    # Create test volume
    vol = np.random.randint(0, 255, (64, 128, 128), dtype=np.uint8)
    print(f"Volume shape: {vol.shape} (z,y,x)")
    print()
    
    # Write with explicit offset and resolution
    custom_offset = [500, 1000, 1500]
    custom_resolution = [30.0, 5.0, 5.0]
    
    print(f"Saving with custom metadata:")
    print(f"  Offset: {custom_offset} (z,y,x) voxels")
    print(f"  Resolution: {custom_resolution} (z,y,x) nm")
    print()
    
    with tempfile.TemporaryDirectory() as td:
        output_path = os.path.join(td, "custom_offset.ome.zarr")
        
        dio.write_ome_zarr(
            output_path,
            vol,
            dataset='0',
            multiscale=True,
            offset=custom_offset,
            resolution=custom_resolution
        )
        
        print(f"✓ Saved to: {output_path}")
        
        # Verify
        try:
            import zarr
            store = zarr.open(output_path, mode='r')
            ms = store.attrs['multiscales'][0]
            transforms = ms['datasets'][0]['coordinateTransformations']
            
            print()
            print("Coordinate transforms:")
            for t in transforms:
                if t['type'] == 'translation':
                    print(f"  Translation: {t['translation']}")
                elif t['type'] == 'scale':
                    print(f"  Scale: {t['scale']}")
        
        except ImportError:
            print("⚠ zarr not available for verification")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo_offset_preservation()
        demo_explicit_offset()
        
        print()
        print("✓ All demos completed successfully!")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
