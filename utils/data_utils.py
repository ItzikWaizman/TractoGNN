import os
import glob
import torch
import numpy as np
from dipy.reconst.shm import sph_harm_lookup
from nibabel import streamlines
import torch.nn.functional as F


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

    # Extract fodf
    fodf_folder = os.path.join(subject_folder, "fodf")
    fodf_path = glob.glob(os.path.join(fodf_folder, "*fodf*"))[0]

    # Extract fractional anisotropy
    fa_folder = os.path.join(subject_folder, "dti")
    fa_path = glob.glob(os.path.join(fa_folder, "*fa*"))[0]

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
        "sh": sh_path,
        "fodf" : fodf_path,
        "fa" : fa_path
    }


def sample_signal_from_sh(data_sh, sh_order, sphere):
    sph_harm_basis = sph_harm_lookup.get("tournier07")
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    Ba = torch.tensor(Ba, dtype=torch.float32)

    X, Y, Z, C = data_sh.shape
    data_sh_reshaped = data_sh.reshape(-1, C)

    data_sampled = torch.matmul(data_sh_reshaped, Ba.T)
    data_sampled = data_sampled.reshape(X,Y,Z, -1)
    return data_sampled


def ras_to_voxel(ras_coords, inverse_affine):
    # Convert ras_coords from NumPy array to PyTorch tensor
    ras_coords_tensor = ras_coords if isinstance(ras_coords, torch.Tensor) else torch.tensor(ras_coords, dtype=torch.float32)
    
    # Append a column of ones for homogeneous coordinates
    ones_column = torch.ones((ras_coords_tensor.shape[0], 1), dtype=torch.float32)
    ras_homogeneous = torch.cat((ras_coords_tensor, ones_column), dim=1)

    # Apply inverse affine transformation to convert RAS to voxel coordinates
    voxel_coords_homogeneous = torch.matmul(ras_homogeneous, inverse_affine.T)

    # Remove homogeneous coordinate and round to get voxel indices
    voxel_coords = torch.round(voxel_coords_homogeneous[:, :3]).to(torch.int)
    return voxel_coords


def voxel_to_ras(voxel_indices, affine):
    # Add a 1 for homogeneous coordinates
    ones_column = torch.ones(voxel_indices.shape[0], 1, dtype=torch.float32)
    voxel_homogeneous = torch.cat((voxel_indices, ones_column), dim=1)

    # Perform affine transformation
    ras_coords_homogeneous = torch.matmul(affine, voxel_homogeneous.T).T

    ras_coords = ras_coords_homogeneous[:, :3]
    return ras_coords


def load_tractogram(tractography_folder):
    folder_path = tractography_folder

    # Get a list of all .trk files in the specified folder
    trk_files = [file for file in os.listdir(folder_path) if file.endswith(".trk")]

    merged_streamlines = []
    # Iterate over the .trk files and merge them
    for trk_file in trk_files:
        current_tractogram = streamlines.load(os.path.join(folder_path, trk_file))
        merged_streamlines.extend(current_tractogram.streamlines)

    return merged_streamlines


def prepare_streamlines_for_training(subject, save_dir_name="torch_streamlines", save_filename="streamlines.pt"):
    """
    Prepares and saves streamlines for training, or loads them if already saved.

    Parameters:
    - subject - SubjectDataHandler object.
    - save_dir: Directory where the tensor and lengths will be saved.
    - save_filename: Filename for saving the tensor and lengths.

    Returns:
    - padded_streamlines: torch tensor of padded streamlines
    """

    save_dir = os.path.join(subject.paths_dictionary['tractography_folder'], save_dir_name)
    save_path = os.path.join(save_dir, save_filename)

    if os.path.exists(save_path):
        data = torch.load(save_path)
        return data['padded_streamlines_tensor'], data['lengths']

    # If the data does not exist, proceed to load and process the streamlines
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare streamlines and lengths
    np_streamlines = load_tractogram(subject.paths_dictionary['tractography_folder'])
    torch_streamlines = []
    for streamline in np_streamlines:
        torch_streamline = torch.tensor(streamline, dtype=torch.float32)
        torch_streamlines.append(torch_streamline)

    max_seq_len = max(len(sl) for sl in torch_streamlines)
    padded_streamlines = torch.zeros(len(torch_streamlines) + 1,max_seq_len, 3)

    for i, streamline in enumerate(torch_streamlines):
        length = len(streamline)
        padded_streamlines[i, :length, :] = streamline

    lengths = torch.tensor([len(sl) for sl in torch_streamlines], dtype=torch.int)

    # Save padded_streamlines and lengths
    torch.save({'padded_streamlines_tensor': padded_streamlines, 'lengths': lengths}, save_path)
    
    return padded_streamlines, lengths


def generate_labels(streamline, actual_length, sphere_points, EoF, sigma=0.1):
    # Truncate the streamline to the actual length + 1 (The streamlines are padded enough to ensure it is safe)
    padded_length = streamline.size(0)
    streamline = streamline[:actual_length+1]
    
    # Calculate direction unit vectors between consecutive points
    directions = streamline[1:] - streamline[:-1]
    directions = directions / directions.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity for all direction vectors with all sphere points
    # directions: (actual_length, 3), sphere_points: (725, 3)
    # cosine_similarity: (actual_length, 725)
    cosine_similarity = torch.matmul(directions, sphere_points.t())
    
    # Convert cosine similarity to distance on the unit sphere
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    distances = torch.acos(cosine_similarity)

    # Apply gaussian kernel
    gaussian_weights = torch.exp(-distances**2 / (2 * sigma**2))
    gaussian_weights[:actual_length-1, 724] = 0 # Set the probability of EoF to zero for any point on the streamline except the last one.

    # Generate FODFs
    fodfs = gaussian_weights / gaussian_weights.sum(dim=1, keepdim=True)
    fodfs[actual_length-1, :] = EoF # Set the fodf of the last point to be the fodf of EoF

    fodfs_padded = torch.zeros((padded_length, 725))
    fodfs_padded[:actual_length, :] = fodfs

    return fodfs_padded