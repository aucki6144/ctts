import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # print("Shape mask: {}".format(mask.shape))
        # print("Shape attn: {}".format(attn.shape))
        if mask is not None:
            if mask.shape == attn.shape:
                attn = attn.masked_fill(mask, -np.inf)
            else:
                uni_size = mask.shape[2]
                attn_expanded = attn.expand(-1, -1, uni_size)  # Expand attn to [32, 41, 41]
                attn_expanded = attn_expanded.masked_fill(mask, -np.inf)
                attn = attn_expanded.mean(dim=2, keepdim=True)  # Reduce it back to [32, 41, 1]

        attn = self.softmax(attn)

        output = torch.bmm(attn, v)

        return output, attn


