import torch
import nibabel as nib
from nibabel import streamlines
from utils.data_utils import *
class SubjectDataHandler(object):
    def __init__(self, subject_folder):
        self.paths_dictionary = extract_subject_paths(subject_folder)
        self.wm_mask = self.load_mask()
        self.tractogram = self.load_tractogram()
        self.dwi, self.affine = self.load_dwi()
        self.dwi_means = self.calc_means(self.dwi)

    def load_dwi(self):
        sh_data = nib.load(self.paths_dictionary['sh'])
        affine = sh_data.affine
        sh_data = torch.tensor(sh_data.get_fdata(), dtype=torch.float32)
        dwi_data = 255 * resample_dwi(sh_data)
        return dwi_data, affine

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

