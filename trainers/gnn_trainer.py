import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from config import *


class GNNTrainer(object):
    """
    Implements a trainer class. All network trainers inherit from this class.
    """

    def __init__(self):
        self._initialize_network()
        self._deep_learning_setup()
        self._initialize_dataloader()

    def _initialize_network(self):
        """
        Every trainer must have some base graph neural network
        """
        self.network = None

    def _deep_learning_setup(self):
        """
        Set up the optimizer and loss criterion
        """
        self.optimizer = Adam(self.network.parameters(), lr=LR)
        self.criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    def calc_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Every trainer must have some loss calculation
        """
        pass

    def _initialize_dataloader(self):
        """
        Set up the data loader - a generator from which we draw batches of graphs
        """
        self.train_dataloader = None
        self.test_dataloader = None

    def forward(self, input_graph: Data) -> torch.Tensor:
        """
        Every trainer must have some forward pass for its network
        :param input_graph: input graph in `torch_geometric.data` format
        :return network outputs for `input_graph`
        """
        pass

    def evaluate(self) -> float:
        """
        Test the model's accuracy
        :return accuracy of the model
        """
        pass

    def train_iteration(self, input_graphs: list[Data], gt_graphs: list[Data]) -> float:
        """
        Single training iteration on a batch of graphs
        :param input_graphs: list of input graphs in `torch_geometric.data` format
        :param gt_graphs: list of ground truth graphs in `torch_geometric.data` format
        :return: loss in the current iteration
        """
        pass
