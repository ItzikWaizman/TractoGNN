import torch
from config import *


def get_link_labels(edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,...,0,0,0,..] with the number of ones is equal to the length of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    length = edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(length, dtype=torch.float, device=DEVICE)
    link_labels[:edge_index.size(1)] = 1.
    return link_labels
