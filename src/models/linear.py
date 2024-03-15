import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            query (Tensor): 
            key (Tensor): _description_
            value (Tensor): _description_
            mask (Optional[Tensor], optional): _description_. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn
    
class AttentionEncoder(nn.Module):
    def __init__(self, in_features, query_dim, out_features, num_hidden_units, atten_mode="temporal"):
        super().__init__()
        self.atten_mode = atten_mode
        
        self.fc_key = nn.Linear(in_features, num_hidden_units)
        self.fc_value = nn.Linear(in_features, num_hidden_units)
        self.fc_query = nn.Linear(query_dim, num_hidden_units)
        
        
        self.fc1 = nn.Linear(num_hidden_units, num_hidden_units)
        self.fc2 = nn.Linear(num_hidden_units, out_features)
        # self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        self.attn = ScaledDotProductAttention(num_hidden_units)

    def forward(self, x, query):
        """_summary_

        Args:
            x (_type_): [batch_sz, seq_len, n_fts]
            query (_type_): [batch_sz, hid_dim]

        Returns:
            _type_: _description_
        """
        if self.atten_mode == "feature":
            x = x.permute(0, 2, 1)
        x_value = self.fc_key(x)
        x_key = self.fc_value(x)
        x_query = self.fc_query(query)
        
        output, _ = self.attn(x_query, x_key, x_value)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output


