import nibabel as nib
from dipy.data import get_sphere
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import *
from utils.common_utils import *

# Data Handler Modes
TRAIN, VALIDATION, TRACK = 0, 1, 2

class SubjectDataHandler(object):
    def __init__(self, logger, params, mode):
        logger.info(f"Create SubjectDataHandler object with mode {mode}")
        self.mode = mode
        self.logger = logger
        self.paths_dictionary = extract_subject_paths(self.get_subject_folder(mode, params))
        self.wm_mask = self.load_mask()
        self.dwi, self.fodf, self.affine, self.inverse_affine, self.fa_map = self.load_subject_data(mode)

        self.logger.info("SubjectDataHandler: Preparing streamlines")
        self.tractogram, self.lengths = prepare_streamlines_for_training(self)
        
        if mode is TRAIN or mode is VALIDATION or TRACK:
            self.data_loader = self.create_dataloaders(batch_size=params['batch_size'])
            self.causality_mask  = torch.triu(torch.ones(self.tractogram.size(1), self.tractogram.size(1)), diagonal=1).bool()
        
    def get_subject_folder(self, mode, params):
        if mode == TRAIN:
            return params['train_subject_folder']
        elif mode == VALIDATION:
            return params['val_subject_folder']
        else:
            return params['test_subject_folder']

    def load_subject_data(self, mode):
        dwi_sh = nib.load(self.paths_dictionary['sh'])
        fodf_sh = nib.load(self.paths_dictionary['fodf'])

        fa_map = nib.load(self.paths_dictionary['fa'])
        fa_map = torch.tensor(fa_map.get_fdata(), dtype=torch.float32) if mode is TRACK else None

        affine = torch.tensor(dwi_sh.affine, dtype=torch.float32)
        dwi_sh_data = torch.tensor(dwi_sh.get_fdata(), dtype=torch.float32)
        fodf_sh_data = torch.tensor(fodf_sh.get_fdata(), dtype=torch.float32)

        dwi_data = sample_signal_from_sh(dwi_sh_data, sh_order=6, sphere=get_sphere('repulsion100'))
        fodf_data = sample_signal_from_sh(fodf_sh_data, sh_order=8, sphere=get_sphere('repulsion724')) if mode is TRACK else None
        return dwi_data, fodf_data, affine, torch.inverse(affine), fa_map


    def load_mask(self):
        mask_path = self.paths_dictionary['wm_mask']
        mask = nib.load(mask_path)
        mask = torch.tensor(mask.get_fdata(), dtype=torch.float32)
        return mask
    
    def create_dataloaders(self, batch_size):
        dataset = StreamlineDataset(self.tractogram, self.lengths, self.inverse_affine, self.mode)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return data_loader

class StreamlineDataset(Dataset):
    def __init__(self, streamlines, lengths, inverse_affine, mode):
        permutation = torch.arange(0, streamlines.size(0)-1)
        permutation = permutation[torch.randperm(permutation.size(0))]

        self.streamlines = streamlines #[permutation[0:500]] if mode is TRAIN else streamlines[permutation[0:200]]
        self.lengths = lengths #[permutation[0:500]] if mode is TRAIN else lengths[permutation[0:200]]

        sphere = get_sphere('repulsion724')
        self.sphere_points = torch.zeros((725, 3), dtype=torch.float32)
        self.sphere_points[:724, :] = torch.tensor(sphere.vertices)

        EoF = torch.zeros(725, dtype=torch.float32)
        EoF[724] = 1
        self.EoF = EoF
        self.inverse_affine = inverse_affine

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        """
        Parameters:
        - idx (int): Index of the streamline to fetch.
            
        Returns:
        - tuple: (streamline, label, seq_length, padding_mask)
        """
        streamline = self.streamlines[idx]
        streamline_voxels = ras_to_voxel(streamline, inverse_affine=self.inverse_affine)
        seq_length = self.lengths[idx]
        label = generate_labels(streamline, seq_length, self.sphere_points, self.EoF)
        padding_mask = torch.arange(streamline.size(0)) >= seq_length
        return streamline_voxels, label, seq_length, padding_mask
