import torch
import torch.nn as nn
import torch_geometric.transforms as transforms
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges


class GCNAutoencoder(nn.Module):
    def __init__(self, feature_dimensions):
        super(GCNAutoencoder, self).__init__()
        self.num_layers = len(feature_dimensions) - 1
        self.conv_layers = nn.ModuleList([GCNConv(feature_dimensions[i], feature_dimensions[i+1])
                                          for i in range(self.num_layers)])
        self.activation = nn.ReLU()

    def encode(self, input_graph):
        x = input_graph.x
        edge_index = input_graph.edge_index
        for layer_idx in range(self.num_layers):
            x = self.activation(self.conv_layers[layer_idx](x, edge_index))
        return x

    @staticmethod
    def decode(z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    @staticmethod
    def decode_all(z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
