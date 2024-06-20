import torch
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np

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



def plot_distribution_on_sphere_dipy(sphere, intensity):
    vertices = sphere.vertices
    faces = sphere.faces

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    i1 = faces[:, 0]
    i2 = faces[:, 1]
    i3 = faces[:, 2]

    # Normalize intensity to be between 0 and 1
    intensity = np.array(intensity)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            colorbar=dict(title='f(x, y, z)', tickvals=[0, np.max(intensity)]),
            colorscale='Jet',  # Change to a heatmap-like colorscale
            intensity=intensity,
            i=i1,
            j=i2,
            k=i3,
            name='y',
            showscale=True,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.5),
            flatshading=False
        )])
    
    plot(fig, filename='sphere_distribution.html', auto_open=True)
