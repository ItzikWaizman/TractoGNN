import os
import glob
import torch
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sph_harm_lookup

def extract_subject_paths(subject_folder):
    # Check if the subject folder exists
    if not os.path.exists(subject_folder):
        print(f"Error: Subject folder not found - {subject_folder}")
        return None

    # Extract bvals, bvecs, and dwi_data paths
    dwi_folder = os.path.join(subject_folder, "dwi")
    dwi_data_path = glob.glob(os.path.join(dwi_folder, "*dwi.nii*"))[0]

    # Extract white matter mask path
    mask_folder = os.path.join(subject_folder, "mask")
    wm_mask_path = glob.glob(os.path.join(mask_folder, "*mask_wm*"))[0]

    # Extract tractography folder path
    tractography_folder = os.path.join(subject_folder, "tractography")

    # Extract spherical harmonics path
    sh_folder = os.path.join(subject_folder, "sh")
    sh_path = glob.glob(os.path.join(sh_folder, "*sh.nii*"))[0]

    # Return the extracted paths
    return {
        "dwi_data": dwi_data_path,
        "wm_mask": wm_mask_path,
        "tractography_folder": tractography_folder,
        "sh": sh_path
    }

def resample_dwi(data_sh):
    # Resamples a diffusion signal according to a set of directions using spherical harmonics.
    sphere = get_sphere('repulsion100')
    sph_harm_basis = sph_harm_lookup.get("tournier07")

    # TractInferno dataset includes spherical harmonics coefficients up to order 6.
    sh_order = 6

    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    Ba = torch.tensor(Ba, dtype=torch.float32)
    data_resampled = torch.matmul(data_sh, Ba.t())
    return data_resampled

def ras_to_voxel(ras_coords, affine):
    inverse_affine = np.linalg.inv(affine)

    # Add a 1 for homogeneous coordinates
    ras_homogeneous = np.append(ras_coords, 1)
    voxel_coords_homogeneous = np.dot(inverse_affine, ras_homogeneous)

    # Remove the homogeneous coordinate and round to get voxel indices
    voxel_coords = np.round(voxel_coords_homogeneous[:3]).astype(int)
    return voxel_coords

def voxel_to_ras(voxel_indices, affine):
    # Add a 1 for homogeneous coordinates
    voxel_homogeneous = np.append(voxel_indices, 1)
    ras_coords_homogeneous = np.dot(affine, voxel_homogeneous)

    # The result is already in RAS+ space; no need to round
    ras_coords = ras_coords_homogeneous[:3]
    return ras_coords
