import torch    

def create_attention_mask(sequence_lengths, max_seq_len):
    # Convert sequence_lengths to a mask tensor
    # Attention mask should be of shape [batch_size, 1, seq_len, seq_len]
    batch_size = len(sequence_lengths)
    mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len) < torch.tensor(sequence_lengths).unsqueeze(1)
    mask = mask.to(dtype=torch.bool)
    # Invert the mask for compatibility with src_key_padding_mask which expects 'True' for positions to ignore
    return ~mask