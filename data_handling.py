import nibabel as nib
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import *


class SubjectDataHandler(object):
    def __init__(self, logger, params, train):
        logger.info("Create SubjectDataHandler object")
        self.logger = logger
        self.train = train
        self.paths_dictionary = extract_subject_paths(params['train_subject_folder'] if self.train else
                                                      params['val_subject_folder'])
        self.wm_mask = self.load_mask()
        self.index_to_voxel, self.voxel_to_index = self.get_voxel_index_maps()
        self.dwi, self.affine, self.inverse_affine = self.load_dwi()
        self.logger.info("SubjectDataHandler: Prepare streamlines for training")
        self.tractogram, self.lengths = prepare_streamlines_for_training(self)
        self.data_loader = self.create_dataloaders(batch_size=params['batch_size'])
        self.graph = self.create_connected_graph(params['train_subject_folder'] if self.train else
                                                 params['val_subject_folder'])
        self.casuality_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.tractogram.size(1))
        
    def get_voxel_index_maps(self):
        # Create a map between node index to white matter voxel.
        index_to_voxel = torch.nonzero(self.wm_mask)
        # Create the inverse map between white matter voxel to node index.
        voxel_to_index = {(row[0].item(), row[1].item(), row[2].item()): i for i, row in enumerate(index_to_voxel)}

        return index_to_voxel, voxel_to_index

    def load_dwi(self):
        sh_data = nib.load(self.paths_dictionary['sh'])
        affine = sh_data.affine
        sh_data = torch.tensor(sh_data.get_fdata(), dtype=torch.float32)
        dwi_data = resample_dwi(sh_data)
        dwi_means, dwi_stds = self.calc_means(dwi_data)
        dwi_data = (dwi_data - dwi_means) / dwi_stds
        # original_dwi_data = nib.load(self.paths_dictionary["dwi_data"])
        # original_dwi_data = torch.tensor(original_dwi_data.get_fdata(), dtype=torch.float32)
        return dwi_data, affine, np.linalg.inv(affine)

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

    def create_connected_graph(self, subject_folder, radius=1, filter_connections=False):
        self.logger.info("SubjectDataHandler: Create connected graph")
        graph_dir = os.path.join(subject_folder, "graph")
        graph_path = os.path.join(graph_dir, "connected_graph.pt")
        if os.path.exists(graph_path):
            # Load and return the graph
            return torch.load(graph_path)

        wm_mask = self.wm_mask
        voxel_indices = self.index_to_voxel
        position_to_index = self.voxel_to_index

        # Create node feature matrix
        node_ras_coordinates = voxel_to_ras(voxel_indices, torch.tensor(self.affine, dtype=torch.float32))
        node_feature_matrix = torch.cat((self.dwi[wm_mask == 1], node_ras_coordinates), dim=1)

        # Create Edge matrix
        edge_index_set = set()
        for row, i in position_to_index.items():

            dims = [np.linspace(row[j] - radius, row[j] + radius, 2 * radius + 1) for j in range(3)]
            mesh = np.meshgrid(*dims)
            neighbors = np.concatenate((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1),
                                        mesh[2].reshape(-1, 1)), axis=1).astype(int)
            target_nodes = [position_to_index.get(tuple(neighbor), -1) for neighbor in neighbors]
            target_nodes = [index for index in target_nodes if index != -1]

            if target_nodes:
                src_nodes = [i for _ in target_nodes]
                edge_index_set.update(zip(src_nodes, target_nodes))

        if filter_connections:
            edge_index_set = filter_tuples(edge_index_set)

        edge_index = torch.tensor(list(edge_index_set)).T
        graph = Data(x=node_feature_matrix, edge_index=edge_index)

        # Ensure the directory exists
        os.makedirs(graph_dir, exist_ok=True)
        # Save the graph
        torch.save(graph, graph_path)

        return graph
    
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
        Args:
            idx (int): Index of the streamline to fetch.
            
        Returns:
            tuple: (streamline, label, seq_length) where label is generated for the streamline.
        """
        streamline = self.streamlines[idx]
        streamline_voxels = self.index_to_voxel[streamline.long()]
        seq_length = self.lengths[idx]
        label = generate_labels_for_streamline(streamline_voxels, self.relative_positions)
        bool_mask = torch.arange(streamline.size(0)) >= seq_length
        return streamline, label, seq_length, bool_mask
