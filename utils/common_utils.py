import torch

def generate_unit_sphere_points(num_points=1000):
    """
    Generate points on the unit sphere using repulsion method in PyTorch.
    """
    indices = torch.arange(0, num_points, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / num_points)
    theta = torch.pi * (1 + torch.sqrt(torch.tensor(5.0))) * indices
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)

def voxel_to_ras(voxel_indices, affine):
    # Add a 1 for homogeneous coordinates
    ones_column = torch.ones(voxel_indices.shape[0], 1, dtype=torch.float32)
    voxel_homogeneous = torch.cat((voxel_indices, ones_column), dim=1)

    # Perform affine transformation
    ras_coords_homogeneous = torch.matmul(affine, voxel_homogeneous.T).T

    ras_coords = ras_coords_homogeneous[:, :3]
    return ras_coords

def ras_to_voxel(ras_coords, inverse_affine):
    # Convert ras_coords from NumPy array to PyTorch tensor
    ras_coords_tensor = torch.tensor(ras_coords, dtype=torch.float32)
    
    # Append a column of ones for homogeneous coordinates
    ones_column = torch.ones((ras_coords_tensor.shape[0], 1), dtype=torch.float32)
    ras_homogeneous = torch.cat((ras_coords_tensor, ones_column), dim=1)

    # Apply inverse affine transformation to convert RAS to voxel coordinates
    voxel_coords_homogeneous = torch.matmul(ras_homogeneous, inverse_affine.T)

    # Remove homogeneous coordinate and round to get voxel indices
    voxel_coords = torch.round(voxel_coords_homogeneous[:, :3]).to(torch.int)
    return voxel_coords