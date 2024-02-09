import os
import glob
import numpy as np
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

def extract_subject_paths(subject_folder):
    # Check if the subject folder exists
    if not os.path.exists(subject_folder):
        print(f"Error: Subject folder not found - {subject_folder}")
        return None

    # Extract bvals, bvecs, and dwi_data paths
    dwi_folder = os.path.join(subject_folder, "dwi")
    bval_path = glob.glob(os.path.join(dwi_folder, "*.bval"))[0]
    bvec_path = glob.glob(os.path.join(dwi_folder, "*.bvec"))[0]
    dwi_data_path = glob.glob(os.path.join(dwi_folder, "*dwi.nii*"))[0]

    # Extract white matter mask path
    mask_folder = os.path.join(subject_folder, "mask")
    wm_mask_path = glob.glob(os.path.join(mask_folder, "*mask_wm*"))[0]

    # Extract tractography folder path
    tractography_folder = os.path.join(subject_folder, "tractography")

    # Extract spherical harmonics path
    sh_folder = os.path.join(subject_folder, "sh")
    sh_path = glob.glob(os.path.join(sh_folder, "*sh.nii*"))

    # Return the extracted paths
    return {
        "bval": bval_path,
        "bvec": bvec_path,
        "dwi_data": dwi_data_path,
        "wm_mask": wm_mask_path,
        "tractography_folder": tractography_folder,
        "sh": sh_path
    }

def normalize_dwi(weights, bvals):
    """ Normalize dwi by the first b0.
    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    Returns
    -------
    ndarray
        Diffusion weights normalized by the B0.
    """
    b0_idx = np.where(bvals == np.min(bvals))
    b0 = weights[..., b0_idx].mean(axis=3) + 1e-10
    b0 = b0[..., None]

    # Make sure in every voxels weights are lower than ones from the b0.
    nb_erroneous_voxels = np.sum(weights > b0)
    if nb_erroneous_voxels != 0:
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed

def resample_dwi(dwi, data_sh):
    # Resamples a diffusion signal according to a set of directions using spherical harmonics.
    sphere = get_sphere('repulsion100')
    sph_harm_basis = sph_harm_lookup.get("tournier07")

    # Tract inferno dataset includes spherical harmonics coefficients up to order 6.
    sh_order = 6

    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, Ba.T)
    return data_resampled