import os
import glob
import torch
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup
from nibabel import streamlines


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


def ras_to_voxel(ras_coords, inverse_affine):
    # Append a column of ones for homogeneous coordinates
    ones_column = np.ones((ras_coords.shape[0], 1), dtype=np.float32)
    ras_homogeneous = np.hstack((ras_coords, ones_column))

    # Apply inverse affine transformation to convert RAS to voxel coordinates
    voxel_coords_homogeneous = np.dot(ras_homogeneous, inverse_affine.T)

    # Remove homogeneous coordinate and round to get voxel indices
    voxel_coords = np.round(voxel_coords_homogeneous[:, :3]).astype(int)
    return voxel_coords


def voxel_to_ras(voxel_indices, affine):
    # Add a 1 for homogeneous coordinates
    ones_column = torch.ones(voxel_indices.shape[0], 1, dtype=torch.float32)
    voxel_homogeneous = torch.cat((voxel_indices, ones_column), dim=1)

    # Perform affine transformation
    ras_coords_homogeneous = torch.matmul(affine, voxel_homogeneous.T).T

    ras_coords = ras_coords_homogeneous[:, :3]
    return ras_coords


def filter_tuples(tuples_set):
    filtered_set = set()

    for tup in tuples_set:
        reverse_tup = (tup[1], tup[0])

        if tup not in filtered_set and reverse_tup not in filtered_set and tup[0] != tup[1]:
            filtered_set.add(tup)

    return filtered_set


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
    processed_streamlines = []
    for streamline in np_streamlines:
        streamline_voxels = ras_to_voxel(streamline, subject.inverse_affine)
        streamline_indices = [subject.voxel_to_index.get(tuple(pos.tolist()), -1) for pos in streamline_voxels]
        streamline_indices = [index for index in streamline_indices if index != -1]
        processed_streamlines.append(streamline_indices)

    max_seq_len = max(len(sl) for sl in processed_streamlines)
    padded_streamlines = [sl + [0]*(max_seq_len - len(sl)) for sl in processed_streamlines]
    padded_streamlines_tensor = torch.tensor(padded_streamlines, dtype=torch.int)
    lengths = torch.tensor([len(sl) for sl in processed_streamlines], dtype=torch.int)

    # Save padded_streamlines_tensor and actual_lengths
    torch.save({'padded_streamlines_tensor': padded_streamlines_tensor, 'lengths': lengths}, save_path)
    
    return padded_streamlines_tensor, lengths


def generate_labels_for_streamline(streamline, rel_positions, cube_size=9):
    """
    Calculate distances from voxels within a 9x9x9 cube centered around each voxel in a streamline
    to the next voxel in the streamline, then apply softmax to generate probability distribution.
    
    Parameters:
    - streamline: Tensor of shape [n, 3] containing voxel indices (x, y, z) for the streamline.
    - cube_size: The edge length of the cube; default is 9, indicating a 9x9x9 cube.
    
    Returns:
    -  probabilities: Tensor of shape [n, 730] containing probability distributions,
                      including for the "end of fiber" class..
    """
    # Replicate the streamline positions to match the number of relative positions
    expanded_streamline = streamline[:-1].unsqueeze(1).expand(-1, cube_size**3, -1)
    
    # Calculate the positions of all voxels within the cubes
    voxel_positions = expanded_streamline + rel_positions
    
    # Calculate differences between voxel positions and the next point in the streamline
    next_voxel_diffs = voxel_positions - streamline[1:].unsqueeze(1)
    
    # Compute Euclidean distances
    distances = torch.norm(next_voxel_diffs.float(), dim=2)
    probabilities = torch.softmax(-torch.pow(distances, 2), dim=-1)

    # Append column of zeros to indicate this is not the end of the fiber
    end_of_fiber_column = torch.zeros(probabilities.shape[0], 1)
    probabilities = torch.cat([probabilities, end_of_fiber_column], dim=-1)

    # Append row of zeros and 1 in the last entry to indicate end of row with probability 1.
    end_of_fiber_row = torch.zeros(1, cube_size**3+1)
    end_of_fiber_row[-1] = 1

    probabilities = torch.cat([probabilities, end_of_fiber_row], dim=0)
    return probabilities
