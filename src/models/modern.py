"""Modern time-series architectures for I-1 (compact re-implementations, CPU-sized).

All map x (B, L, 1 [+extra broadcast]) -> (B, 180, n_bands) in log10 space.
  PatchTST   (Nie et al. 2023)  : patching + Transformer encoder + flatten head, with a
                                  level-aware instance normalisation (input mean removed, then
                                  added back through a learned per-band gain).
  TSMixer    (Chen et al. 2023) : alternating time-mixing / feature-mixing MLPs.
  NBEATS     (Oreshkin et al. 2020): generic stacks of FC blocks with backcast/forecast.
  ModernTCN  (Luo & Wang 2024)  : depth-wise large-kernel conv + point-wise FFN blocks (lite).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_OUT = 180


class LevelHead(nn.Module):
    """y = core + g_b * level + c_b, level = mean of the input window (log10)."""

    def __init__(self, n_bands):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n_bands))
        self.c = nn.Parameter(torch.zeros(n_bands))

    def forward(self, core, level):  # core (B,180,nb), level (B,1)
        return core + level[:, :, None] * self.g[None, None, :] + self.c[None, None, :]


class PatchTST(nn.Module):
    def __init__(self, n_in=60, n_bands=4, patch=8, stride=4, d_model=64, nhead=4, num_layers=2,
                 dropout=0.1, extra_in=0):
        super().__init__()
        self.patch, self.stride = patch, stride
        n_p = (n_in - patch) // stride + 1
        self.embed = nn.Linear(patch + extra_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_p, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_p * d_model, N_OUT * n_bands)
        self.level = LevelHead(n_bands)
        self.n_bands = n_bands

    def forward(self, x, extra=None):
        s = x[:, :, 0]
        lvl = s.mean(1, keepdim=True)
        z = (s - lvl).unfold(1, self.patch, self.stride)            # (B, n_p, patch)
        if extra is not None:
            z = torch.cat([z, extra[:, None, :].expand(-1, z.size(1), -1)], -1)
        h = self.enc(self.embed(z) + self.pos)
        core = self.head(self.drop(h.flatten(1))).view(-1, N_OUT, self.n_bands)
        return self.level(core, lvl)


class MixerBlock(nn.Module):
    def __init__(self, L, C, hidden, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.time = nn.Sequential(nn.Linear(L, L), nn.GELU(), nn.Dropout(dropout))
        self.feat = nn.Sequential(nn.Linear(C, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, C),
                                  nn.Dropout(dropout))

    def forward(self, h):  # (B, L, C)
        h = h + self.time(self.ln1(h).transpose(1, 2)).transpose(1, 2)
        return h + self.feat(self.ln2(h))


class TSMixer(nn.Module):
    def __init__(self, n_in=60, n_bands=4, channels=16, hidden=64, num_blocks=2, dropout=0.1, extra_in=0):
        super().__init__()
        self.inp = nn.Linear(1 + extra_in, channels)
        self.blocks = nn.Sequential(*[MixerBlock(n_in, channels, hidden, dropout) for _ in range(num_blocks)])
        self.tproj = nn.Linear(n_in, N_OUT)
        self.cproj = nn.Linear(channels, n_bands)
        self.level = LevelHead(n_bands)

    def forward(self, x, extra=None):
        lvl = x[:, :, 0].mean(1, keepdim=True)
        z = x - lvl[:, :, None]
        if extra is not None:
            z = torch.cat([z, extra[:, None, :].expand(-1, z.size(1), -1)], -1)
        h = self.blocks(self.inp(z))                               # (B, L, C)
        h = self.tproj(h.transpose(1, 2)).transpose(1, 2)          # (B, 180, C)
        return self.level(self.cproj(h), lvl)


class NBeatsBlock(nn.Module):
    def __init__(self, n_in, n_out, width, layers, dropout):
        super().__init__()
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.fc = nn.Sequential(*mods)
        self.back = nn.Linear(width, n_in)
        self.fore = nn.Linear(width, n_out)

    def forward(self, x):
        h = self.fc(x)
        return self.back(h), self.fore(h)


class NBEATS(nn.Module):
    def __init__(self, n_in=60, n_bands=4, width=256, layers=4, num_blocks=3, dropout=0.1, extra_in=0):
        super().__init__()
        self.n_in, self.n_bands, self.extra_in = n_in, n_bands, extra_in
        self.blocks = nn.ModuleList([NBeatsBlock(n_in + extra_in, N_OUT * n_bands, width, layers, dropout)
                                     for _ in range(num_blocks)])
        self.level = LevelHead(n_bands)

    def forward(self, x, extra=None):
        s = x[:, :, 0]
        lvl = s[:, -1:]
        r = s - lvl
        fc = 0
        for blk in self.blocks:
            inp = r if extra is None else torch.cat([r, extra], 1)
            b, f = blk(inp)
            r = r - b[:, : self.n_in]
            fc = fc + f
        return self.level(fc.view(-1, N_OUT, self.n_bands), lvl)


class TCNBlock(nn.Module):
    def __init__(self, C, kernel, ffn, dropout):
        super().__init__()
        self.dw = nn.Conv1d(C, C, kernel, padding=kernel // 2, groups=C)
        self.bn = nn.BatchNorm1d(C)
        self.ffn = nn.Sequential(nn.Conv1d(C, ffn * C, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(ffn * C, C, 1),
                                 nn.Dropout(dropout))

    def forward(self, h):
        return h + self.ffn(self.bn(self.dw(h)))


class ModernTCN(nn.Module):
    def __init__(self, n_in=60, n_bands=4, channels=32, kernel=31, ffn=2, num_blocks=2, dropout=0.1, extra_in=0):
        super().__init__()
        self.inp = nn.Conv1d(1 + extra_in, channels, 1)
        self.blocks = nn.Sequential(*[TCNBlock(channels, kernel, ffn, dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(channels * n_in, N_OUT * n_bands)
        self.drop = nn.Dropout(dropout)
        self.level = LevelHead(n_bands)
        self.n_bands = n_bands

    def forward(self, x, extra=None):
        lvl = x[:, :, 0].mean(1, keepdim=True)
        z = x - lvl[:, :, None]
        if extra is not None:
            z = torch.cat([z, extra[:, None, :].expand(-1, z.size(1), -1)], -1)
        h = self.blocks(self.inp(z.transpose(1, 2)))
        core = self.head(self.drop(h.flatten(1))).view(-1, N_OUT, self.n_bands)
        return self.level(core, lvl)


def build_modern(cfg: dict, n_in: int = 60, n_bands: int = 4, extra_in: int = 0) -> nn.Module:
    kind = cfg["kind"]
    kw = {k: v for k, v in cfg.items() if k not in ("kind", "lr", "batch_size", "bands", "weight_decay")}
    cls = {"patchtst": PatchTST, "tsmixer": TSMixer, "nbeats": NBEATS, "moderntcn": ModernTCN}[kind]
    return cls(n_in=n_in, n_bands=n_bands, extra_in=extra_in, **kw)
