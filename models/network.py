import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from utils.model_utils import *


class TractoTransformer(nn.Module):
    def __init__(self, logger, params):
        super(TractoTransformer, self).__init__()
        logger.info("Create TractoGNN model object")

        # Build decoder-only transformer model
        self.positional_encoding = PositionalEncoding(d_model=params['num_of_gradients'], max_len=params['max_streamline_len'])
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(embed_dim=params['num_of_gradients'],
                                                                     num_heads=params['nhead'],
                                                                     ff_dim=params['transformer_feed_forward_dim'],
                                                                     dropout=params['dropout_rate']) for _ in range(params['num_transformer_decoder_layers'])])
        
        # Use Linear network as a projection to output_size.
        self.projection = nn.Linear(params['num_of_gradients'], params['output_size'])
        self.dropout = nn.Dropout(params['dropout_rate'])

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the fully connected layer
        init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            init.zeros_(self.projection.bias)
        
        # Initialize the transformer decoder layer weights
        for layer in self.decoder_layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    init.xavier_uniform_(param)
        
    def forward(self, dwi_data, streamline_voxels_batch, padding_mask, causality_mask):
        """
        Parameters:
        - dwi_data - 4d image holding the diffusion mri data.
        - streamline_voxels_batch - Tensor of shape [batch_size, max_sequence_length, 3], holds the streamlines coordinates in voxel space.
        - sequence_lengths - the actual lengths of strimlines in the batch.
        - padding_mask - boolean mask of shape [batch_size, max_sequence_length, 1] with 'True' where the values are zero padded.
        - casuality_mask - Triangular matrix so the attention will not take into account future steps.

        Returns:
        - probabilities: Tensor of shape [batch_size, max_sequence_length, 725] modeling the estimated fodfs.
        """       
        # Fetch the features of each point in the streamlines:
        y = dwi_data[streamline_voxels_batch[:, :, 0], streamline_voxels_batch[:, :, 1], streamline_voxels_batch[:, :, 2]]

        # Apply positional encoding
        x = self.dropout(self.positional_encoding(y))
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, causality_mask, padding_mask)

        outputs = self.projection(x)

        return outputs

        # Normalize phi to be in the range [-pi, pi]
        #phi = torch.tanh(outputs[..., 0]) * torch.pi

        # Normalize theta to be in the range [0, pi]
        #theta = torch.sigmoid(outputs[..., 1]) * torch.pi

        #return torch.stack((phi, theta), dim=-1)
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff = PositionWiseFeedForward(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causality_mask, padding_mask):
        # Self-attention with masking
        attn_output, _ = self.self_attn(x, x, x, attn_mask=causality_mask, key_padding_mask=padding_mask, is_causal=True)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-forward network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()
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
        return x + self.encoding[:x.size(1), :].to(x.device).unsqueeze(0)
    