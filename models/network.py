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
        self.embedding = nn.Embedding(3600000, params['num_of_gradients'])
        self.positional_encoding = PositionalEncoding(d_model=params['num_of_gradients'], max_len=params['max_streamline_len'])
        
        decoder_layer = TransformerDecoderLayer(d_model=params['num_of_gradients'],
                                                nhead=params['nhead'],
                                                dim_feedforward=params['transformer_feed_forward_dim'],
                                                dropout=params['dropout_rate'],
                                                batch_first=True)
        
        self.decoder = TransformerDecoder(decoder_layer, params['num_transformer_decoder_layers'])
        
        
        # Use Linear network as a projection to output_size.
        self.projection = nn.Linear(params['num_of_gradients'], params['output_size'])
        self.dropout = nn.Dropout(params['dropout_rate'])


        
    def forward(self, dwi_data, streamline_voxels_batch, streamline_ids, padding_mask, causality_mask):
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
        tgt_embeddings = self.embedding(streamline_ids)
        memory = dwi_data[streamline_voxels_batch[:, :, 0], streamline_voxels_batch[:, :, 1], streamline_voxels_batch[:, :, 2]]

        # Apply positional encoding
        x = self.dropout(self.positional_encoding(tgt_embeddings))
        
        x = self.decoder(tgt=tgt_embeddings, memory=memory, tgt_mask=causality_mask, memory_mask=causality_mask,
                         tgt_key_padding_mask=padding_mask, memory_key_padding_mask=padding_mask)

        outputs = self.projection(x)

        return outputs

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
    