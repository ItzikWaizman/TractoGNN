import torch.nn as nn
import nibabel as nib
from utils.model_utils import *
from utils.data_utils import *
from utils.common_utils import *

class DebugModel(nn.Module):
    def __init__(self, fodf, affine, inverse_affine):
        super(DebugModel, self).__init__()
        self.fodf, self.affine, self.inverse_affine = fodf, affine, inverse_affine
           
    def forward(self, streamlines, padding_mask, step):
        ras_coords = streamlines[:, step, :]
        voxel_coords = ras_to_voxel(ras_coords, self.inverse_affine)
        x_coords, y_coords, z_coords = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]

        fodf_result = self.fodf[x_coords, y_coords, z_coords]
        fodf_result = torch.nn.functional.softmax(fodf_result, dim=-1)

        zeros = torch.zeros(fodf_result.size(0), 1, dtype=fodf_result.dtype, device=fodf_result.device)
        returned_fodf = torch.cat((fodf_result, zeros), dim=-1)
        
        return returned_fodf

