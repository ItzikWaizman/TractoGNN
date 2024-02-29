import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from utils.trainer_utils import FEATURE_DIMS


class GCNAutoencoder(nn.Module):
    def __init__(self, feature_dimensions):
        super(GCNAutoencoder, self).__init__()
        self.num_layers = len(feature_dimensions) - 1
        self.conv_layers = nn.ModuleList([GCNConv(feature_dimensions[i], feature_dimensions[i+1])
                                          for i in range(self.num_layers)])
        self.activation = nn.ReLU()
        self.rel_emb = Parameter(torch.empty(1, FEATURE_DIMS[-1]))
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def encode(self, input_graph):
        x = input_graph.x
        edge_index = input_graph.edge_index
        for layer_idx in range(self.num_layers):
            x = self.activation(self.conv_layers[layer_idx](x, edge_index))
        return x

    def decode(self, z, edge_index):
        logits = (z[edge_index[0]] * self.rel_emb * z[edge_index[1]]).sum(dim=-1)
        return logits

