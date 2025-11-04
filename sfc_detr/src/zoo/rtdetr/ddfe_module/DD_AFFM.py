import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, Concat, RepC3
from .AWFM import AdaptiveWindowFrequencyModulation


class FAM(nn.Module):
    def __init__(self, dim, e=3):
        super().__init__()
        self.e = e
        self.cv2 = Conv(dim // e, dim // e, 1)
        self.m = AdaptiveWindowFrequencyModulation(dim=dim // e, window_size=4)
        self.conv_543 = Conv(dim, dim // e, 1)

    def forward(self, x):
        conv543_ = self.conv_543(x)
        return self.cv2(self.m(conv543_) + conv543_)

class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, stride=2):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.linear = nn.Linear(in_channels * patch_size * patch_size, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.transpose(1, 2)

        fuse = self.linear(patches)
        out = self.norm(fuse)

        H_out, W_out = H // self.stride, W // self.stride
        return out.transpose(1, 2).reshape(B, -1, H_out, W_out)

class HFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pd = PatchDown(in_channels, out_channels)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        c = x
        m = x

        c = self.pd(c)
        x = self.conv(x)
        x = self.batch_norm1(x)

        m = self.max(m)
        m = self.batch_norm2(m)
        return x + c + m

class BiCAF(nn.Module):
    def __init__(self, hidden_dim, istop2bottom) -> None:
        super().__init__()
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

        self.tail_conv = Conv(hidden_dim, hidden_dim, 1)
        self.conv_1x1_fuse = Conv(hidden_dim * 2, hidden_dim, 1)
        self.istop2bottom = istop2bottom

    def forward(self, input):
        low_f, high_f = input
        batch_size, channel, w, h = low_f.shape
        fuse = self.conv_1x1_fuse(torch.cat(input, dim=1))
        d = self.d.expand(batch_size, channel, w, h)
        score = torch.sigmoid(d)
        f = score * low_f + (1 - score) * high_f

        if self.istop2bottom:
            return self.tail_conv(f + fuse) + high_f
        else:
            return self.tail_conv(f + fuse) + low_f

class DD_AFFM(nn.Module):

    def __init__(self, hidden_dim, e):
        super().__init__()

        self.REPC3 = nn.ModuleList([
            nn.Sequential(
                RepC3(hidden_dim, hidden_dim, e=0.5),
                RepC3(hidden_dim, hidden_dim, e=e),
                RepC3(hidden_dim, hidden_dim, e=0.5)
            ) for _ in range(4)
        ])

        self.upsample_54 = nn.Upsample(None, 2, 'nearest')
        self.fuse_54 = BiCAF(hidden_dim, True)

        """ P3 """
        self.upsample_43 = nn.Upsample(None, 2, 'nearest')
        self.downsample_23 = HFD(hidden_dim, hidden_dim)
        self.cat_543 = Concat()
        self.fam = FAM(hidden_dim * 3, 3)

        """ P4 """
        self.downsample_34 = PatchDown(hidden_dim, hidden_dim)
        
        self.fuse_34 = BiCAF(hidden_dim, False)

        """ P5 """
        self.downsample_45 = PatchDown(hidden_dim, hidden_dim)
        self.fuse_45 = BiCAF(hidden_dim, False)

    def forward(self, feats):
        """ ---------------top->bottom--------------- """

        """ 5->4 """
        UP_54 = self.upsample_54(feats[-1])
        F_54 = self.fuse_54([UP_54, feats[-2]])
        R_54 = self.REPC3[0](F_54)

        """ 4->3 && 2->3 """
        UP_43 = self.upsample_43(R_54)
        DOWN_23 = self.downsample_23(feats[0])
        CAT_543 = self.cat_543([DOWN_23, UP_43, feats[1]])
        FAM = self.fam(CAT_543)
        R_543 = self.REPC3[1](FAM)

        """ ---------------bottom->top--------------- """

        """ 3->4 """
        DOWN_34 = self.downsample_34(R_543)
        F_34 = self.fuse_34([DOWN_34, R_54])
        R_34 = self.REPC3[2](F_34)

        """ 4->5 """
        DOWN_45 = self.downsample_45(R_34)
        F_45 = self.fuse_45([DOWN_45, feats[-1]])
        R_45 = self.REPC3[3](F_45)

        return [R_543, R_34, R_45]
