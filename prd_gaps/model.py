"""Mask-aware multiscale encoder, active multihead attention, signal decoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class PartialConv(nn.Module):
    def __init__(self, cin, cout, stride=4):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 9, stride=stride, padding=4, bias=False)
        self.bias = nn.Parameter(torch.zeros(cout))
        self.stride = stride
        self.norm = nn.GroupNorm(min(8, cout), cout)

    def forward(self, x, valid):
        count = F.conv1d(valid, torch.ones(1, 1, 9, device=x.device, dtype=x.dtype),
                         stride=self.stride, padding=4)
        support = count > 0
        y = self.conv(x*valid)*9/count.clamp_min(1)+self.bias[None, :, None]
        y = F.gelu(self.norm(y))*support
        return y, support.to(x.dtype)

class GapReconstructor(nn.Module):
    """10 feature channels; separate A/E masks are features 4 and 5.

    Four stride-4 stages keep long-context attention affordable. Invalid input
    tokens are excluded as keys; their query positions infer missing features.
    """
    def __init__(self, widths=(24, 48, 96, 128), layers=4, heads=4):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 10
        for width in widths:
            self.enc.append(PartialConv(cin, width)); cin = width
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=widths[-1], nhead=heads, dim_feedforward=4*widths[-1],
            dropout=.1, activation='gelu', batch_first=True, norm_first=True),
            num_layers=layers, enable_nested_tensor=False)
        self.dec = nn.ModuleList()
        current = widths[-1]
        for width in reversed(widths[:-1]):
            self.dec.append(nn.Sequential(nn.Conv1d(current+width, width, 5, padding=2),
                                          nn.GroupNorm(8, width), nn.GELU()))
            current = width
        self.out = nn.Sequential(nn.Conv1d(current+10, 48, 7, padding=3), nn.GELU(),
                                 nn.Conv1d(48, 2, 7, padding=3))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] != 10:
            raise ValueError('Expected B x 10 x N features')
        valid = features[:, 4:6].amax(dim=1, keepdim=True)
        if (valid.sum(dim=-1) == 0).any():
            raise ValueError('Each record needs observed samples')
        x = features; skips = []
        for block in self.enc:
            x, valid = block(x, valid); skips.append(x)
        tokens = x.transpose(1, 2)
        length, dim = tokens.shape[1:]
        positions = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        rates = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=x.dtype)*(-math.log(10000)/dim))
        pos = torch.zeros(length, dim, device=x.device, dtype=x.dtype)
        pos[:, 0::2] = torch.sin(positions*rates); pos[:, 1::2] = torch.cos(positions*rates)
        x = self.attention(tokens+pos, src_key_padding_mask=~valid[:, 0].bool()).transpose(1, 2)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        x = F.interpolate(x, size=features.shape[-1], mode='linear', align_corners=False)
        return self.out(torch.cat((x, features), dim=1))

def supervised_loss(prediction, target, valid, psd, floor_fraction=1e-4):
    """Supervised clean-target loss. The spectral term is a distance, not a likelihood."""
    energy = target.square().mean(dim=(-2, -1)).clamp_min(1e-8)
    error = (prediction-target).square()
    full = error.mean(dim=(-2, -1))/energy
    missing = 1-valid
    gap = (error*missing).sum(dim=(-2, -1))/missing.sum(dim=(-2, -1)).clamp_min(1)/energy
    floor = psd[:, 1:].median(dim=-1).values[:, None]*floor_fraction
    weight = torch.maximum(psd, floor).reciprocal().clone()
    weight[:, 0] = 0
    weight[:, -1] *= .5
    p = torch.fft.rfft(prediction.float(), dim=-1)
    t = torch.fft.rfft(target.float(), dim=-1)
    spectral = ((p-t).abs().square()*weight).sum(dim=(-2, -1)) / \
               (t.abs().square()*weight).sum(dim=(-2, -1)).clamp_min(1e-8)
    # Logarithmic robustification bounds the influence of very weak signals
    # without dropping them or using SNR as an inference-time input.
    return (torch.log1p(full)+.5*torch.log1p(gap)+.25*torch.log1p(spectral)).mean()
