import random
import numpy as np
import nibabel as nib
from nibabel import streamlines
from dipy.io import read_bvals_bvecs
from utils.data_utils import *

class SubjectDataHandler(object):
    def __init__(self, subject_folder):
        self.paths_dictionary = extract_subject_paths(subject_folder)
        self.dwi = self.load_dwi()
        self.wm_mask = self.load_mask()
        self.tractogram = self.load_tractogram()
        self.bvals, self.bvecs = read_bvals_bvecs(self.paths_dictionary['bval'], self.paths_dictionary['bvec'])
        self.dwi_means = self.calc_means(self.dwi)

    def load_dwi(self):
        dwi_data = nib.load(self.paths_dictionary['dwi_data']).get_fdata().astype("float32")
        sh_data = nib.load(self.paths_dictionary['sh']).get_fdata().astype("float32")

        dwi_data = normalize_dwi(dwi_data, self.bvals)
        dwi_data = resample_dwi(dwi_data, sh_data)

        return dwi_data

    def calc_means(self, dwi):
        # Reshape the dwi tensor to make it suitable for vectorized operations
        dwi_reshaped = dwi.view(-1, dwi.shape[3])

        # Apply mask if available
        dwi_masked = dwi_reshaped[self.wm_mask > 0]

        # Calculate mean along the 0th dimension
        DW_means = np.mean(dwi_masked, dim=0)
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
        dwi_data = nib.load(mask_path)
        return dwi_data.get_fdata().astype("float32")

