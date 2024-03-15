
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.model_utils import *

class TractoGNN(nn.Module):
    def __init__(self, logger, params):
        super(TractoGNN, self).__init__()

        # Fetch configurations
        self.graph_encoder_feature_dims = params['graph_encoder_feature_dims']
        self.latent_space_dim =  self.graph_encoder_feature_dims[-1]
        self.dropout_rate = params['dropout_rate']
        self.max_seq_len = params['max_streamline_len']
        self.nhead = params['nhead']
        self.transformer_forward_dim = params['transformer_feed_forward_dim']
        self.num_transformer_encoder_layers = params['num_transformer_encoder_layers']
        self.output_size = params['output_size']

        # Set graph autoencoder layer
        self.graph_encoder = GCNAutoEncoder(self.graph_encoder_feature_dims, self.dropout_rate)

        # Set positional encoding layer
        self.positional_encoding = PositionalEncoding(self.latent_space_dim, self.dropout_rate, self.max_seq_len)

        # Define TransformerEncoderLayers
        encoder_layer = TransformerEncoderLayer(d_model=self.latent_space_dim,
                                                 nhead=self.nhead,
                                                 dim_feedforward=self.transformer_forward_dim,
                                                 dropout=self.dropout_rate,
                                                 batch_first=True) 
           
        # Wrap the TransformerEncoderLayers with TrnasformerEncoder
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_transformer_encoder_layers)
        
        # Use Linear network as a decoder.
        self.decoder = nn.Linear(self.latent_space_dim, self.output_size)
        
    def forward(self, input_graph, node_sequence_batch, sequence_lengths):
        """
        node_sequence_batch - Tensor of shape [seqence_len, batch_size].
        sequence_lengths - the actual lengths of any sequence in the batch.

        """        
        # Encode the entire graph
        graph_node_features = self.graph_encoder.encode(input_graph)  # Shape: [num_nodes, feature_dim]
        
        # Extract features for the current sequence of nodes. node_sequence_batch assumed to be padded with 0s, meaning 
        # values that exceeds the corresponding sequence length are not valid.
        node_features_batch = graph_node_features[node_sequence_batch] 
        # Apply positional encoding
        node_features_batch = self.positional_encoding(node_features_batch) 
        
        # Process sequence with TransformerEncoder
        attention_mask = create_attention_mask(sequence_lengths, node_features_batch.size(0))
        encoded_sequence = self.transformer_encoder(node_features_batch, src_key_padding_mask=attention_mask) 
        
        output_sequences = self.decoder(encoded_sequence)
        probabilities = F.log_softmax(output_sequences, dim=-1)

        return probabilities


class GCNAutoEncoder(nn.Module): 
    def __init__(self, encoder_feature_dims, dropout_rate):
        super(GCNAutoEncoder, self).__init__()

        # Number of GCN layers
        self.num_layers = len(encoder_feature_dims) - 1
        self.conv_layers = nn.ModuleList([GCNConv(encoder_feature_dims[i],  encoder_feature_dims[i+1]) for i in range(self.num_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def encode(self, input_graph):
        # x - graph node features. Tensor of shape #num_of_nodex X (#mri_gradient_directions + concatination of 3 RAS coordinates specifying geometric location).
        x, edge_index = input_graph.x, input_graph.edge_index

        for layer_idx in range(self.num_layers):
            x = self.conv_layers[layer_idx](x, edge_index)
            if layer_idx < self.num_layers - 1:  # Apply dropouts before last layer
                x = self.activation(x)
                x = self.dropout(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to input tensor x.
        
        Parameters:
            x (Tensor): Input tensor of shape [batch_size, seq_length, d_model].
        
        Returns:
            Tensor: The input tensor augmented with positional encodings.
        """
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)