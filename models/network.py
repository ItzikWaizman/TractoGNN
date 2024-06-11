import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.model_utils import *


class TractoGNN(nn.Module):
    def __init__(self, logger, params):
        super(TractoGNN, self).__init__()
        logger.info("Create TractoGNN model object")
        # Fetch configurations
        self.logger = logger
        self.latent_space_dim = params['num_of_gradients']
        self.dropout_rate = params['dropout_rate']
        self.max_seq_len = params['max_streamline_len']
        self.nhead = params['nhead']
        self.transformer_forward_dim = params['transformer_feed_forward_dim']
        self.num_transformer_encoder_layers = params['num_transformer_encoder_layers']
        self.output_size = params['output_size']

        # Set positional encoding layer
        self.positional_encoding = PositionalEncoding(self.latent_space_dim, self.dropout_rate, self.max_seq_len)

        # Define TransformerEncoderLayers
        encoder_layer = TransformerEncoderLayer(d_model=self.latent_space_dim,
                                                nhead=self.nhead,
                                                dim_feedforward=self.transformer_forward_dim,
                                                dropout=self.dropout_rate,
                                                batch_first=True)
           
        # Wrap the TransformerEncoderLayers with TransformerEncoder
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_transformer_encoder_layers)
        
        # Use Linear network as a decoder.
        self.decoder = nn.Sequential(nn.Linear(self.latent_space_dim, self.output_size))
        
    def forward(self, dwi_data, node_sequence_batch, padding_mask, casuality_mask):
        """
        Parameters:
        - dwi_data - 4d image holding the diffusion mri data.
        - node_sequence_batch - Tensor of shape [batch_size, max_sequence_length, 3], holds the streamlines coordinates in voxel space.
        - sequence_lengths - the actual lengths of strimlines in the batch.
        - padding_mask - boolean mask of shape [batch_size, max_sequence_length, 1] with 'True' where the values are zero padded.
        - casuality_mask - Triangular matrix so the attention will not take into account future steps.

        Returns:
        - probabilities: Tensor of shape [batch_size, max_sequence_length, 725] modeling the estimated fodfs.
        """       
        # Fetch the features of each point in the streamlines:
        x = dwi_data[node_sequence_batch[:, :, 0], node_sequence_batch[:, :, 1], node_sequence_batch[:, :, 2]]

        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Process sequence with TransformerEncoder
        encoded_sequence = self.transformer_encoder(x,
                                                    mask=casuality_mask,
                                                    src_key_padding_mask=padding_mask)

        output_sequences = self.decoder(encoded_sequence)
        probabilities = F.log_softmax(output_sequences, dim=-1)

        return probabilities

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float
        div_term = (10000 ** (_2i / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        """
        Adds positional encoding to input tensor x.
        
        Parameters:
            x (Tensor): Input tensor of shape [batch_size, seq_length, d_model].
        
        Returns:
            Tensor: The input tensor augmented with positional encodings.
        """
        return x + self.encoding[:, :].to(x.device).unsqueeze(0)
