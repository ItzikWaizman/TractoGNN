import torch
import numpy as np
import nibabel as nib
import time
from torch.nn.functional import conv3d
from nibabel import streamlines
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from utils.data_utils import *

class SubjectDataHandler(object):
    def __init__(self, subject_folder):
        #TODO: Do not hold wm_mask, tractogram and dwi on RAM. All we need is the graph representation.
        self.paths_dictionary = extract_subject_paths(subject_folder)
        self.wm_mask = self.load_mask()
        self.tractogram = self.load_tractogram()
        self.dwi, self.affine, self.inverse_affine = self.load_dwi()
        self.dwi_means = self.calc_means(self.dwi)

    def load_dwi(self):
        sh_data = nib.load(self.paths_dictionary['sh'])
        affine = sh_data.affine
        sh_data = torch.tensor(sh_data.get_fdata(), dtype=torch.float32)
        dwi_data = 255 * resample_dwi(sh_data)
        return dwi_data, affine, np.linalg.inv(affine)

    def calc_means(self, dwi):
        DW_means = torch.zeros(dwi.shape[3])
        mask = self.wm_mask

        for i in range(len(DW_means)):
            curr_volume = dwi[:, :, :, i]
            curr_volume = curr_volume[mask > 0]
            DW_means[i] = torch.mean(curr_volume)

        return DW_means

    def load_tractogram(self):
        folder_path = self.paths_dictionary['tractography_folder']

        # Get a list of all .trk files in the specified folder
        trk_files = [file for file in os.listdir(folder_path) if file.endswith(".trk")]

        merged_streamlines = []
        # Iterate over the .trk files and merge them
        for trk_file in trk_files:
            current_tractogram = streamlines.load(os.path.join(folder_path, trk_file))
            merged_streamlines.extend(current_tractogram.streamlines)

        return merged_streamlines

    def load_mask(self):
        mask_path = self.paths_dictionary['wm_mask']
        mask = nib.load(mask_path)
        mask = torch.tensor(mask.get_fdata(), dtype=torch.float32)
        return mask

    def expand_white_matter_mask(self, white_matter_mask, radius=1):
        # Define a 3D kernel with a cube shape to capture neighboring voxels within the specified radius
        kernel_size = 2 * radius + 1
        kernel = torch.ones(kernel_size, kernel_size, kernel_size, dtype=torch.bool)

        # Dilate the white matter mask using the kernel to expand it
        expanded_mask = conv3d(white_matter_mask.unsqueeze(0).unsqueeze(0).float(),
                               kernel.unsqueeze(0).unsqueeze(0).float(),
                               padding=radius).bool().squeeze()

        return expanded_mask

    def create_ground_truth_graph(self):
        # Create node feature matrix
        #TODO: Decide if we should expand white matter mask. wm_mask = self.expand_white_matter_mask(self.wm_mask)
        wm_mask = self.wm_mask
        node_feature_matrix = self.dwi[wm_mask == 1]

        # Create a map between row index (node index) and the corresponding ras coordinates
        voxel_indices = torch.nonzero(wm_mask)
        position_to_index = {(row[0].item(), row[1].item(), row[2].item()): i for i, row in enumerate(voxel_indices)}

        # Create Edge matrix
        edge_index_set = set()
        for streamline in self.tractogram:

            streamline_voxels = ras_to_voxel(streamline, self.inverse_affine)
            streamline_indices = [position_to_index.get(tuple(pos.tolist()), -1) for pos in streamline_voxels]
            streamline_indices = [index for index in streamline_indices if index != -1]

            src_nodes = streamline_indices[:-1]
            target_nodes = streamline_indices[1:]
            edge_index_set.update(zip(src_nodes, target_nodes))

        edge_index = torch.tensor(list(edge_index_set)).T
        return Data(x=node_feature_matrix, edge_index=edge_index)