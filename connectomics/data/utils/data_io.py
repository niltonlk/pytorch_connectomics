from __future__ import print_function, division
from typing import Optional, List, Union

# Avoid PIL "IOError: image file truncated"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import h5py
import math
import glob
import numpy as np
import imageio
import pickle
from scipy.ndimage import zoom

# Optional imports for additional volume formats
try:  # OME-Zarr
    import zarr  # type: ignore
except Exception:  # pragma: no cover - optional dep
    zarr = None  # type: ignore

try:  # Neuroglancer precomputed via CloudVolume
    from cloudvolume import CloudVolume  # type: ignore
except Exception:  # pragma: no cover - optional dep
    CloudVolume = None  # type: ignore


def readimg_as_vol(filename, drop_channel=False):
    img_suf = filename[filename.rfind('.')+1:]
    assert img_suf in ['png', 'tif']
    data = imageio.imread(filename)

    if data.ndim == 3 and not drop_channel:
        # convert (y,x,c) to (c,y,x) shape
        data = data.transpose(2,0,1)
        return data
    
    elif drop_channel and data.ndim == 3:
        # convert RGB image to grayscale by average
        data = np.mean(data, axis=-1).astype(np.uint8)

    return data[np.newaxis, :, :]   # return data as (1,y,x) shape


def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        # load the first dataset in the h5 file
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def readvol(filename: str, dataset: Optional[str]=None, drop_channel: bool=False):
    r"""Load volumetric data in HDF5, TIFF or PNG formats.
    """
    # Dispatch by prefix/suffix for extended formats
    if filename.startswith('precomputed://') or filename.startswith('gs://') \
       or filename.startswith('s3://') or filename.startswith('file://'):
        return readvol_precomputed(filename, roi_spec=dataset, drop_channel=drop_channel)

    if filename.endswith('.zarr') or filename.endswith('.ome.zarr'):
        return readvol_ome_zarr(filename, dataset=dataset, drop_channel=drop_channel)

    img_suf = filename[filename.rfind('.')+1:]
    if img_suf in ['h5', 'hdf5']:
        data = readh5(filename, dataset)
    elif 'tif' in img_suf:
        data = imageio.volread(filename).squeeze()
        if data.ndim == 4:
            # convert (z,c,y,x) to (c,z,y,x) order
            data = data.transpose(1,0,2,3)
    elif 'png' in img_suf:
        data = readimgs(filename)
        if data.ndim == 4:
            # convert (z,y,x,c) to (c,z,y,x) order
            data = data.transpose(3,0,1,2)
    else:
        raise ValueError('unrecognizable file format for %s' % (filename))

    assert data.ndim in [3, 4], "Currently supported volume data should " + \
        "be 3D (z,y,x) or 4D (c,z,y,x), got {}D".format(data.ndim)
    if drop_channel and data.ndim == 4:
        # merge multiple channels to grayscale by average
        orig_dtype = data.dtype
        data = np.mean(data, axis=0).astype(orig_dtype)
 
    return data


###############################
# Extended readers
###############################

def _parse_roi(roi: Optional[str]) -> Optional[List[slice]]:
    """Parse ROI spec like "x0:x1,y0:y1,z0:z1" into slices for CloudVolume.

    CloudVolume uses x,y,z ordering (Fortran-style). Accepts separators comma,
    semicolon, or pipe between axes. Returns None if roi is None.
    """
    if roi is None:
        return None
    # Allow URL anchors like ...#x0:x1,y0:y1,z0:z1
    if '#' in roi:
        roi = roi.split('#', 1)[1]
    # Normalize separators to commas
    roi = roi.replace('|', ',').replace(';', ',')
    parts = [p.strip() for p in roi.split(',') if p.strip()]
    if len(parts) != 3:
        raise ValueError("ROI must have three parts: 'x0:x1,y0:y1,z0:z1' (CloudVolume x,y,z order)")

    def _parse_part(p: str) -> slice:
        a, b = p.split(':')
        return slice(int(a), int(b))

    x, y, z = map(_parse_part, parts)
    return [x, y, z]


def _maybe_reorder_channels(arr: np.ndarray, drop_channel: bool=False) -> np.ndarray:
    """Reorder/strip channels to match expected (z,y,x) or (c,z,y,x).

    Heuristics:
    - If 5D, assume (t,c,z,y,x) or (t,z,y,x,c); pick t=0, then reduce to 4D.
    - If 4D and one axis looks like channels (<=10), move to channel-first.
    - If the final axis is singleton channel, squeeze when drop_channel or keep as grayscale.
    """
    data = arr
    # Handle 5D (t, c, z, y, x) or (t, z, y, x, c)
    if data.ndim == 5:
        # choose the first timepoint
        if data.shape[1] <= 10:  # (t,c,z,y,x)
            data = data[0]
        elif data.shape[-1] <= 10:  # (t,z,y,x,c)
            data = data[0]
        else:
            # Assume time first, drop it
            data = data[0]

    if data.ndim == 4:
        # Could be (c,z,y,x) or (z,y,x,c)
        # If first dim small, assume channel-first already
        if data.shape[0] <= 10 and data.shape[1] > 16:
            pass  # (c,z,y,x)
        elif data.shape[-1] <= 10:
            data = data.transpose(3, 0, 1, 2)  # (z,y,x,c) -> (c,z,y,x)
        else:
            # Unknown, assume already (c,z,y,x)
            pass

    elif data.ndim == 3:
        # (z,y,x) OK
        pass

    # If channel dimension exists but is 1 and drop_channel requested, collapse
    if data.ndim == 4 and data.shape[0] == 1 and drop_channel:
        data = data[0]

    return data


def readvol_ome_zarr(path: str, dataset: Optional[str]=None, drop_channel: bool=False) -> np.ndarray:
    """Read an OME-Zarr or .zarr volume.

    Selection priority:
    1) If 'dataset' is provided, use it as a group/array key inside the store.
       - For multiscales/pyramids, use '0' (full res), '1', '2', etc.
       - Example: readvol('data.ome.zarr', dataset='2')  # MIP level 2
    2) If group has OME-NGFF 'multiscales', open datasets[0]['path'] (usually '0').
    3) Otherwise, pick the first array found in a BFS walk.
    """
    if zarr is None:
        raise ImportError("zarr is required to read OME-Zarr volumes. Install 'zarr' and retry.")

    store = zarr.open(path, mode='r')

    def _first_array(node):
        # BFS over group tree to find first zarr array
        queue = [node]
        while queue:
            cur = queue.pop(0)
            try:
                import zarr as _z  # local alias
                if isinstance(cur, _z.Array):
                    return cur
            except Exception:
                pass
            if hasattr(cur, 'values'):
                for v in cur.values():
                    queue.append(v)
        return None

    arr = None
    grp = store
    # Check for multiscales attr at group root
    try:
        attrs = getattr(grp, 'attrs', {})
        ms = attrs.get('multiscales') if hasattr(attrs, 'get') else None
    except Exception:
        ms = None

    if dataset is not None:
        # Support URLs like path#group/key
        key = dataset.split('#', 1)[-1]
        try:
            arr = grp[key]
        except Exception as e:
            raise KeyError(f"Dataset key '{key}' not found in zarr store: {e}")
    elif ms:
        try:
            key = ms[0]['datasets'][0]['path']
            arr = grp[key]
        except Exception:
            arr = _first_array(grp)
    else:
        arr = _first_array(grp)

    if arr is None:
        raise RuntimeError("No array found in the provided OME-Zarr store.")

    data = np.asarray(arr)
    data = _maybe_reorder_channels(data, drop_channel=drop_channel)
    # Final validation
    assert data.ndim in [3, 4], f"OME-Zarr reader expects 3D or 4D arrays, got {data.ndim}D"
    return data


def readvol_precomputed(source: str, roi_spec: Optional[str]=None, drop_channel: bool=False, mip: int=0) -> np.ndarray:
    """Read a Neuroglancer 'precomputed' volume using CloudVolume.

    To avoid accidental massive downloads, an ROI must be provided either:
    - via the 'roi_spec' argument (e.g., '0:64,0:64,0:64' in x,y,z order), or
    - appended as a URL anchor to the source (e.g., 'precomputed://...#0:64,0:64,0:64').
    
    MIP level selection:
    - Use 'mip=0' (default) for full resolution
    - Use 'mip=1', 'mip=2', etc. for downsampled pyramid levels
    - Or append @mip to URL: 'precomputed://...@2#0:64,0:64,0:64'

    Note: CloudVolume uses x,y,z (Fortran) ordering for both slicing and returned arrays.
    This reader transposes to z,y,x (C-order) to match the rest of the pipeline.
    """
    if CloudVolume is None:
        raise ImportError("cloud-volume is required to read 'precomputed' sources. Install 'cloud-volume' and retry.")

    # Parse MIP level from URL if present (e.g., precomputed://...@2#roi)
    url, anchor = source, None
    if '@' in source:
        # Extract MIP from @N notation
        parts = source.split('@')
        url = parts[0]
        rest = parts[1]
        if '#' in rest:
            mip_str, anchor = rest.split('#', 1)
            mip = int(mip_str)
            if roi_spec is None:
                roi_spec = anchor
        else:
            mip = int(rest)
    elif '#' in source:
        url, anchor = source.split('#', 1)
        if roi_spec is None:
            roi_spec = anchor

    roi_slices = _parse_roi(roi_spec)
    if roi_slices is None:
        raise ValueError("ROI is required for 'precomputed' volumes. Provide 'x0:x1,y0:y1,z0:z1' via dataset or URL #anchor.")

    # CloudVolume expects slicing as (x,y,z) and returns (x,y,z[,c])
    cv = CloudVolume(url, mip=mip, progress=False, fill_missing=True, cache=False)
    xsl, ysl, zsl = roi_slices
    vol = cv[xsl, ysl, zsl]  # returns np.ndarray with shape (x,y,z[,c])
    data = np.asarray(vol)

    # Transpose from CloudVolume (x,y,z[,c]) to pipeline standard (z,y,x[,c])
    if data.ndim == 4:
        # (x,y,z,c) -> (z,y,x,c) -> (c,z,y,x)
        data = data.transpose(2, 1, 0, 3)  # (z,y,x,c)
        data = data.transpose(3, 0, 1, 2)  # (c,z,y,x)
        if data.shape[0] == 1 and drop_channel:
            data = data[0]
    elif data.ndim == 3:
        # (x,y,z) -> (z,y,x)
        data = data.transpose(2, 1, 0)
    else:
        raise RuntimeError(f"Unexpected dimensionality from CloudVolume: {data.shape}")

    assert data.ndim in [3, 4]
    return data


def savevol(filename, vol, dataset='main', format='h5'):
    if format == 'h5':
        writeh5(filename, vol, dataset='main')
    if format == 'png':
        currentDirectory = os.getcwd()
        img_save_path = os.path.join(currentDirectory, filename)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        for i in range(vol.shape[0]):
            imageio.imsave('%s/%04d.png' % (img_save_path, i), vol[i])


def readim(filename, do_channel=False):
    # x,y,c
    if not os.path.exists(filename):
        im = None
    else:  # note: cv2 do "bgr" channel order
        im = imageio.imread(filename)
        if do_channel and im.ndim == 2:
            im = im[:, :, None]
    return im


def readimgs(filename):
    filelist = sorted(glob.glob(filename))
    num_imgs = len(filelist)

    # decide numpy array shape:
    img = imageio.imread(filelist[0])
    if img.ndim == 2:
        data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
    elif img.ndim == 3:
        data = np.zeros((num_imgs, img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    data[0] = img

    # load all images
    if num_imgs > 1:
        for i in range(1, num_imgs):
            data[i] = imageio.imread(filelist[i])

    return data

def read_pkl(filename):
    """
    The function `read_pkl` reads a pickle file and returns a list of the objects stored in the file.

    :param filename: The filename parameter is a string that represents the name of the file you want to
    read. It should include the file extension, such as ".pkl" for a pickle file
    :return: a list of objects that were read from the pickle file.
    """
    data = []
    with open(filename, "rb") as fid:
        while True:
            try:
                data.append(pickle.load(fid))
            except EOFError:
                break
    if len(data) == 1:
        return data[0]
    return data

def writeh5(filename, dtarray, dataset='main'):
    fid = h5py.File(filename, 'w')
    if isinstance(dataset, (list,)):
        for i, dd in enumerate(dataset):
            ds = fid.create_dataset(
                dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(dataset, dtarray.shape,
                                compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()


def create_json(ndim: int = 1, dtype: str = "uint8", data_path: str = "/path/to/data/",
                height: int = 10000, width: int = 10000, depth: int = 500,
                n_columns: int = 3, n_rows: int = 3, tile_size: int = 4096,
                tile_ratio: int = 1, tile_st: List[int] = [0, 0]):
    """Create a dictionay to store the metadata for large volumes. The dictionary is
    usually saved as a JSON file and can be read by the TileDataset.

    Args:
        ndim (int, optional): [description]. Defaults to 1.
        dtype (str, optional): [description]. Defaults to "uint8".
        data_pathLstr (str, optional): [description]. Defaults to "/path/to/data".
        height (int, optional): [description]. Defaults to 10000.
        width (int, optional): [description]. Defaults to 10000.
        depth (int, optional): [description]. Defaults to 500.
        tile_ratio (int, optional): [description]. Defaults to 1.
        n_columns (int, optional): [description]. Defaults to 3.
        n_rows (int, optional): [description]. Defaults to 3.
        tile_size (int, optional): [description]. Defaults to 4096.
        tile_ratio (int, optional): [description]. Defaults to 1.
        tile_st (List[int], optional): [description]. Defaults to [0,0].
    """
    metadata = {}
    metadata["ndim"] = ndim
    metadata["dtype"] = dtype

    digits = int(math.log10(depth))+1
    metadata["image"] = [
        data_path + str(i).zfill(digits) + r"/{row}_{column}.png"
        for i in range(depth)]

    metadata["height"] = height
    metadata["width"] = width
    metadata["depth"] = depth

    metadata["n_columns"] = n_columns
    metadata["n_rows"] = n_rows

    metadata["tile_size"] = tile_size
    metadata["tile_ratio"] = tile_ratio
    metadata["tile_st"] = tile_st

    return metadata

####################################################################
# tile to volume
####################################################################


def vast2Seg(seg):
    # convert to 24 bits
    if seg.ndim == 2 or seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        return seg[:, :, 0].astype(np.uint32)*65536 + seg[:, :, 1].astype(np.uint32)*256 + seg[:, :, 2].astype(np.uint32)
    elif seg.ndim == 4:  # n rgb image
        return seg[:, :, :, 0].astype(np.uint32)*65536 + seg[:, :, :, 1].astype(np.uint32)*256 + seg[:, :, :, 2].astype(np.uint32)


def tile2volume(tiles: List[str], coord: List[int], coord_m: List[int], tile_sz: Union[int, List[int]],
                dt: type = np.uint8, tile_st: List[int] = [0, 0], tile_ratio: float = 1.0,
                do_im: bool = True, background: int = 128) -> np.ndarray:
    """Construct a volume from image tiles based on the given volume coordinate.

    Args:
        tiles (List[str]): a list of paths to the image tiles.
        coord (List[int]): the coordinate of the volume to be constructed.
        coord_m (List[int]): the coordinate of the whole dataset with the tiles.
        tile_sz (Union[int, List[int]]): the height and width of the tiles.
        dt (type): data type of the constructed volume. Default: numpy.uint8
        tile_st (List[int]): start position of the tiles. Default: [0, 0]
        tile_ratio (float): scale factor for resizing the tiles. Default: 1.0
        do_im (bool): construct an image volume (apply linear interpolation for resizing). Default: `True`
        background (int): background value for filling the constructed volume. Default: 128
    """
    z0o, z1o, y0o, y1o, x0o, x1o = coord  # region to crop
    z0m, z1m, y0m, y1m, x0m, x1m = coord_m  # tile boundary

    bd = [max(-z0o, z0m), max(0, z1o-z1m), max(-y0o, y0m),
          max(0, y1o-y1m), max(-x0o, x0m), max(0, x1o-x1m)]
    z0, y0, x0 = max(z0o, z0m), max(y0o, y0m), max(x0o, x0m)
    z1, y1, x1 = min(z1o, z1m), min(y1o, y1m), min(x1o, x1m)

    result = background*np.ones((z1-z0, y1-y0, x1-x0), dt)
    tile_sz_y = tile_sz[0] if isinstance(tile_sz, list) else tile_sz
    tile_sz_x = tile_sz[1] if isinstance(tile_sz, list) else tile_sz
    c0 = x0 // tile_sz_x  # floor
    c1 = (x1 + tile_sz_x-1) // tile_sz_x  # ceil
    r0 = y0 // tile_sz_y
    r1 = (y1 + tile_sz_y-1) // tile_sz_y
    for z in range(z0, z1):
        pattern = tiles[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                if r'{row}_{column}' in pattern:
                    path = pattern.format(
                        row=row+tile_st[0], column=column+tile_st[1])
                else:
                    path = pattern
                patch = readim(path, do_channel=True)
                if patch is not None:
                    if tile_ratio != 1:  # im ->1, label->0
                        patch = zoom(
                            patch, [tile_ratio, tile_ratio, 1], order=int(do_im))

                    # last tile may not be of the same size
                    xp0 = column * tile_sz_x
                    xp1 = xp0 + patch.shape[1]
                    yp0 = row * tile_sz_y
                    yp1 = yp0 + patch.shape[0]
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    if do_im:  # image
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a -
                               x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0, 0]
                    else:  # label
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a -
                               x0] = vast2Seg(patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])

    # For chunks touching the border of the large input volume, apply padding.
    if max(bd) > 0:
        result = np.pad(
            result, ((bd[0], bd[1]), (bd[2], bd[3]), (bd[4], bd[5])), 'reflect')
    return result
