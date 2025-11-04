import torch
import torch.nn as nn
from einops import rearrange

class AdaptiveWindowFrequencyModulation(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.weight= nn.Parameter(torch.cat((torch.ones(self.window_size, self.window_size // 2 + 1, dim, 1, dtype=torch.float32),\
        torch.zeros(self.window_size, self.window_size//2 + 1, dim, 1, dtype=torch.float32)), dim=-1))

    def forward(self, x):
        x = rearrange(x, 'b c (w1 p1) (w2 p2) -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        x = x.to(torch.float32)
        x= torch.fft.rfft2(x,dim=(3, 4), norm='ortho')
        weight = torch.view_as_complex(self.weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b c (w1 p1) (w2 p2)')
        return x
