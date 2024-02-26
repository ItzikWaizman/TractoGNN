from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from trainers.gnn_trainer import GNNTrainer
from models.gcn_autoencoder import GCNAutoencoder
from utils.trainer_utils import get_link_labels
from config import *


class GCNAETrainer(GNNTrainer):

    def __init__(self):
        super(GCNAETrainer, self).__init__()

    def _initialize_network(self):
        self.network = GCNAutoencoder(FEATURE_DIMS).to(DEVICE)

    def calc_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _initialize_dataloader(self):
        self.train_dataloader = None
        self.test_dataloader = None

    def forward(self, input_graph: Data) -> torch.Tensor:

        z = self.network.encode(input_graph)
        return z

    def evaluate(self) -> float:
        pass

    def train_iteration(self, input_graphs: list[Data], gt_graphs: list[Data]) -> float:

        assert len(input_graphs) == len(gt_graphs)
        loss = 0
        for input_graph, gt_graph in zip(input_graphs, gt_graphs):

            z = self.forward(input_graph)

            neg_edge_index = negative_sampling(
                edge_index=gt_graph.edge_index,
                num_nodes=gt_graph.num_nodes,
                num_neg_samples=gt_graph.edge_index.size(1))

            link_logits = self.network.decode(z, gt_graph.edge_index, neg_edge_index)
            link_labels = get_link_labels(gt_graph.edge_index, neg_edge_index)

            loss = loss + self.criterion(link_logits, link_labels)

        loss_value = loss.item()
        loss.backward()
        self.optimizer.step()

        return loss_value
