import os
import sys
import time
import numpy as np
import nibabel as nib
import itk
from scipy.ndimage import affine_transform

def timing(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()   # high-resolution timer
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"\t{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

class suppress_stdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def nib_load(path, dtype=np.float32, raw=True, orientation=('A', 'R', 'S'), library = ''):
    """
    Load a NIfTI image with NiBabel, reorient to a standard orientation,
    optionally flip axes and update the affine accordingly.

    Parameters
    ----------
    path : str
        Path to the NIfTI image.
    dtype : np.dtype
        Desired data type.
    raw : bool
        If True, return Nifti1Image object; else return (Nifti1Image, np.array).
    orientation : tuple
        Target orientation (default is ('A', 'R', 'S')).
    flip_axes : tuple
        Axes to flip (0=x, 1=y, 2=z).

    Returns
    -------
    volume : nib.Nifti1Image
        NiBabel image with corrected orientation and affine.
    data : np.array (optional)
        Numpy array of image data if raw=False.
    """
    volume = nib.load(path)
    
    # Reorient image
    wanted_orientation = nib.orientations.axcodes2ornt(orientation)
    current_orientation = nib.orientations.io_orientation(volume.affine)
    if not np.array_equal(current_orientation, wanted_orientation):
        transformation = nib.orientations.ornt_transform(current_orientation, wanted_orientation)
        volume = volume.as_reoriented(transformation)
    
    volume_data = volume.get_fdata().copy().astype(dtype)
    affine = volume.affine.copy()
    
    if library == 'ANTs':
        # Flip axes if requested
        for axis in (0,1,2):
            if axis < 0 or axis > 2:
                raise ValueError("flip_axes must be 0, 1, or 2")
            shift = (volume_data.shape[axis] - 1) * volume.affine[:3, axis]  # use original direction
            affine[:3, 3] += shift
            affine[:, axis] *= -1
            volume_data = np.flip(volume_data, axis=axis)
        affine[0]*=-1
        affine[1]*=-1
    
    # Create new NiBabel image with updated affine
    volume = nib.Nifti1Image(volume_data, affine, volume.header)
    
    if raw:
        return volume
    return volume, volume_data

def rescaled_affine(affine, new_voxel_sizes):
    """
    Compute a new affine that preserves the image orientation (rotation)
    but applies new voxel sizes (scaling).
    """
    R = affine[:3, :3]
    t = affine[:3, 3]
    old_vox = nib.affines.voxel_sizes(affine)

    # Normalize columns of R to unit vectors, then scale by new voxel sizes
    R_unit = R @ np.diag(1 / old_vox)
    R_new = R_unit @ np.diag(new_voxel_sizes)

    new_affine = np.eye(4)
    new_affine[:3, :3] = R_new
    new_affine[:3, 3] = t
    return new_affine

def resample_to_voxel_size(img, target_vox, is_label=False):
    """
    Resample a NIfTI image to a given voxel size using affine_transform.
    Preserves orientation and world alignment.
    
    Parameters
    ----------
    img : nib.Nifti1Image
        Input image.
    target_vox : tuple or np.ndarray
        Desired voxel size (mm).
    is_label : bool
        If True, uses nearest-neighbor interpolation (for segmentation labels).
    """
    data = img.get_fdata().copy()
    affine = img.affine
    orig_vox = nib.affines.voxel_sizes(affine)

    scale = orig_vox / target_vox
    new_affine = rescaled_affine(affine, target_vox)

    # Compute transform from output to input voxel coordinates
    transform = np.linalg.inv(affine) @ new_affine
    matrix = transform[:3, :3]
    offset = transform[:3, 3]

    order = 0 if is_label else 1  # nearest neighbor for labels
    resampled = affine_transform(
        data, matrix=matrix, offset=offset,
        output_shape=np.ceil(np.array(data.shape) * scale).astype(int),
        order=order
    )

    # Ensure integer dtype for labels (avoids float rounding issues)
    if is_label:
        resampled = np.rint(resampled).astype(data.dtype)

    return nib.Nifti1Image(resampled, new_affine)

def pad_or_crop_to_match(fixed_img, moving_img):
    """
    Pad or crop the moving image so it matches the fixed image shape,
    keeping centers aligned in voxel space.
    """
    fixed_data = fixed_img.get_fdata()
    moving_data = moving_img.get_fdata()

    fixed_shape = np.array(fixed_data.shape)

    def pad_or_crop(data, target_shape):
        data_shape = np.array(data.shape)
        diff = target_shape - data_shape
        result = data

        # --- Padding if image is smaller ---
        pad_width = [(max(d // 2, 0), max(d - max(d // 2, 0), 0)) for d in diff]
        if any(d > 0 for d in diff):
            result = np.pad(result, pad_width, mode="constant", constant_values=0)

        # --- Cropping if image is larger ---
        if any(d < 0 for d in diff):
            crop_slices = []
            for i in range(3):
                if diff[i] < 0:
                    start = (-diff[i]) // 2
                    end = start + target_shape[i]
                    crop_slices.append(slice(start, end))
                else:
                    crop_slices.append(slice(0, target_shape[i]))
            result = result[tuple(crop_slices)]

        return result

    new_fixed_data = pad_or_crop(fixed_data, fixed_shape)
    new_moving_data = pad_or_crop(moving_data, fixed_shape)

    new_fixed_img = nib.Nifti1Image(new_fixed_data, fixed_img.affine)
    new_moving_img = nib.Nifti1Image(new_moving_data, moving_img.affine)

    return new_fixed_img, new_moving_img

def center_image_in_fixed_affine(moving_nib, fixed_nib):
    """
    Center the moving image inside the fixed image grid based on world-space centers,
    taking into account both affines (orientation, voxel size, and origin).

    Parameters
    ----------
    moving_nib : nib.Nifti1Image
        Moving image (already resampled to same voxel size).
    fixed_nib : nib.Nifti1Image
        Fixed/reference image.

    Returns
    -------
    centered_moving : nib.Nifti1Image
        Moving image data placed inside fixed image grid, preserving world alignment.
    """

    moving_data = moving_nib.get_fdata()
    fixed_shape = np.array(fixed_nib.shape)
    moving_shape = np.array(moving_data.shape)

    fixed_aff = fixed_nib.affine
    moving_aff = moving_nib.affine

    # --- Step 1: Compute world-space centers of both images ---
    fixed_center_world = nib.affines.apply_affine(fixed_aff, fixed_shape / 2)
    moving_center_world = nib.affines.apply_affine(moving_aff, moving_shape / 2)

    # --- Step 2: Compute translation vector (in world coordinates) ---
    shift_world = fixed_center_world - moving_center_world

    # --- Step 3: Convert world shift to voxel-space shift (in fixed grid) ---
    # inverse of rotation/scaling part of fixed affine
    fixed_voxel_to_world = fixed_aff[:3, :3]
    shift_voxel = np.linalg.inv(fixed_voxel_to_world) @ shift_world

    # --- Step 4: Compute integer voxel offset for array placement ---
    offset_voxel = np.round(shift_voxel).astype(int)

    # --- Step 5: Create empty array and place moving data ---
    centered_data = np.zeros(fixed_shape, dtype=moving_data.dtype)

    # compute valid insertion slices in both fixed and moving arrays
    insert_fixed = []
    insert_moving = []
    for dim in range(3):
        start_fixed = max(0, offset_voxel[dim])
        end_fixed = min(fixed_shape[dim], offset_voxel[dim] + moving_shape[dim])
        start_moving = max(0, -offset_voxel[dim])
        end_moving = start_moving + (end_fixed - start_fixed)
        insert_fixed.append(slice(start_fixed, end_fixed))
        insert_moving.append(slice(start_moving, end_moving))

    centered_data[tuple(insert_fixed)] = moving_data[tuple(insert_moving)]

    # --- Step 6: Keep fixed affine (grid stays the same) ---
    return nib.Nifti1Image(centered_data, fixed_aff, fixed_nib.header)

def pad_or_crop_to_shape(nib_img, target_shape):
    data = nib_img.get_fdata()
    affine = nib_img.affine
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)

    result = np.zeros(target_shape, dtype=data.dtype)
    min_shape = np.minimum(current_shape, target_shape)

    # Center crop/pad
    start_src = (current_shape - min_shape) // 2
    end_src = start_src + min_shape
    start_dst = (target_shape - min_shape) // 2
    end_dst = start_dst + min_shape

    slices_src = tuple(slice(s, e) for s, e in zip(start_src, end_src))
    slices_dst = tuple(slice(s, e) for s, e in zip(start_dst, end_dst))
    result[slices_dst] = data[slices_src]
    return nib.Nifti1Image(result, affine, nib_img.header)

from nibabel.processing import resample_from_to
def match_nii_images(fixed_nib, moving_nib, fixed_seg_nib=None, moving_seg_nib=None):
    """
    Match voxel size, affine, and shape of moving to fixed image.
    Optionally processes corresponding segmentations.
    """
    # --- 1. Compute resampling target ---
    target_affine = fixed_nib.affine
    target_shape = fixed_nib.shape
    target = (target_shape, target_affine)

    # --- 2. Resample moving image and segmentation into fixed space ---
    resampled_moving = resample_from_to(moving_nib, target, order=1)
    
    resampled_moving_seg = None
    if moving_seg_nib is not None:
        resampled_moving_seg = resample_from_to(moving_seg_nib, target, order=0)
        return fixed_nib, fixed_seg_nib, resampled_moving, resampled_moving_seg
    
    return fixed_nib, resampled_moving

def zscore_normalize(volume_nib: nib.nifti1.Nifti1Image, eps=1e-8) -> np.ndarray:
    """Apply zscore normalization and reshape data"""
    # compute mean / std only over non-zero voxels (common for brain MRI)
    volume = volume_nib.get_fdata().copy()
    mask = volume != 0
    if mask.sum() == 0:
        v = volume
        mean = v.mean()
        std = v.std()
    else:
        v = volume[mask]
        mean = v.mean()
        std = v.std()
    if std < eps:
        std = eps
    out = (volume - mean) / std

    # clip extreme values for stability
    out = np.clip(out, -5.0, 5.0)

    out = nib.Nifti1Image(out.astype(np.float32), volume_nib.affine, volume_nib.header)
    return out