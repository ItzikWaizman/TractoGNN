import nibabel as nib
from dipy.data import get_sphere
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import *

# Data Handler Modes
TRAIN, VALIDATION, TRACK = 0, 1, 2

class SubjectDataHandler(object):
    def __init__(self, logger, params, mode):
        logger.info("Create SubjectDataHandler object")
        self.logger = logger
        self.paths_dictionary = extract_subject_paths(self.get_subject_folder(mode, params))
        self.wm_mask = self.load_mask()
        self.dwi, self.fodf, self.affine, self.inverse_affine, self.fa_map = self.load_subject_data()
        
        if mode == TRAIN or mode == VALIDATION:
            self.logger.info("SubjectDataHandler: Prepare streamlines for training")
            self.tractogram, self.lengths = prepare_streamlines_for_training(self)
            self.data_loader = self.create_dataloaders(batch_size=params['batch_size'])
            self.casuality_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.tractogram.size(1))
        
    def get_subject_folder(self, mode, params):
        if mode == TRAIN:
            return params['train_subject_folder']
        elif mode == VALIDATION:
            return params['val_subject_folder']
        else:
            return params['test_subject_folder']

    def get_voxel_index_maps(self):
        # Create a map between node index to white matter voxel.
        index_to_voxel = torch.nonzero(self.wm_mask)
        # Create the inverse map between white matter voxel to node index.
        voxel_to_index = {(row[0].item(), row[1].item(), row[2].item()): i for i, row in enumerate(index_to_voxel)}

        return index_to_voxel, voxel_to_index

    def load_subject_data(self):
        dwi_sh = nib.load(self.paths_dictionary['sh'])
        fodf_sh = nib.load(self.paths_dictionary['fodf'])

        fa_map = nib.load(self.paths_dictionary['fa'])
        fa_map = torch.tensor(fa_map.get_fdata(), dtype=torch.float32)

        affine = torch.tensor(dwi_sh.affine, dtype=torch.float32)
        dwi_sh_data = torch.tensor(dwi_sh.get_fdata(), dtype=torch.float32)
        fodf_sh_data = torch.tensor(fodf_sh.get_fdata(), dtype=torch.float32)

        dwi_data = sample_signal_from_sh(dwi_sh_data, sh_order=6, sphere=get_sphere('repulsion100'))
        fodf_data = sample_signal_from_sh(fodf_sh_data, sh_order=8, sphere=get_sphere('repulsion724'))
        return dwi_data, fodf_data, affine, torch.inverse(affine), fa_map

    def calc_means(self, dwi):
        DW_means = torch.zeros(dwi.shape[3])
        DW_stds = torch.zeros(dwi.shape[3])
        mask = self.wm_mask

        for i in range(len(DW_means)):
            curr_volume = dwi[:, :, :, i]
            curr_volume = curr_volume[mask > 0]
            DW_means[i] = torch.mean(curr_volume)
            DW_stds[i] = torch.std(curr_volume)

        return DW_means, DW_stds

    def load_mask(self):
        mask_path = self.paths_dictionary['wm_mask']
        mask = nib.load(mask_path)
        mask = torch.tensor(mask.get_fdata(), dtype=torch.float32)
        return mask
    
    def create_dataloaders(self, batch_size):
        dataset = StreamlineDataset(self.tractogram, self.lengths, self.index_to_voxel, train=self.train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if self.train else False)
        
        return data_loader

class StreamlineDataset(Dataset):
    def __init__(self, streamlines, lengths, index_to_voxel, cube_size=9, train=True):
        perm = torch.arange(0, streamlines.size(0))
        perm = perm[torch.randperm(perm.size(0))]
        self.streamlines = streamlines[perm[0:100000]] if train else streamlines[perm[0:10000]]
        self.lengths = lengths[perm[0:100000]] if train else lengths[perm[0:10000]]
        self.index_to_voxel = index_to_voxel
        self.cube_size = cube_size
        self.relative_positions = self.calculate_rel_positions()

    def calculate_rel_positions(self):
        offset = self.cube_size // 2
        relative_positions = torch.stack(torch.meshgrid(
            torch.arange(-offset, offset + 1),
            torch.arange(-offset, offset + 1),
            torch.arange(-offset, offset + 1),
            indexing='ij'), dim=-1).reshape(-1, 3).int()
        
        return relative_positions

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        """
        Parameters:
        - idx (int): Index of the streamline to fetch.
            
        Returns:
        - tuple: (streamline, label, seq_length) where label is generated for the streamline.
        """
        streamline = self.streamlines[idx]
        streamline_voxels = self.index_to_voxel[streamline.long()]
        seq_length = self.lengths[idx]
        label = generate_labels_for_streamline(streamline_voxels, self.relative_positions)
        bool_mask = torch.arange(streamline.size(0)) >= seq_length
        return streamline, label, seq_length, bool_mask
