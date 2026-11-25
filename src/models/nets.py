"""Deep-learning models replicating the Zenodo scripts of Kim et al. (2026).

Inputs are log10 flux tensors of shape (B, L_in, 1) (L_in = 60 by default).
Outputs are log10 irradiance of shape (B, 180, n_out_bands).

FCN (band-specific, n_out_bands = 1):
  hidden dims = [int(h*scale) for h in (180, 720, 2880)] arranged by num_expansion
  1 -> [h1]; 2 -> [h1,h2,h1]; 3 -> [h1,h2,h3,h2,h1]; optional BatchNorm1d; Dropout
  (AlphaDropout for SELU). w0 variant uses the "anchor": input = [x, x[-1]] (61),
  output = x[-1] + delta.
RNN/LSTM/GRU (joint 4-band): recurrent stack -> Dropout -> Linear(H, 180*4).
Transformer (joint 4-band): Linear(1,d) -> sinusoidal PE -> encoder xL -> mean-pool -> Linear(d, 720).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

BASE_H = (180, 720, 2880)
N_OUT = 180


def build_hidden_dims(num_expansion: int, scale_factor: float) -> list[int]:
    h1, h2, h3 = [int(h * scale_factor) for h in BASE_H]
    if num_expansion == 1:
        return [h1]
    if num_expansion == 2:
        return [h1, h2, h1]
    return [h1, h2, h3, h2, h1]


def get_activation(name: str) -> nn.Module:
    return {"selu": nn.SELU(), "relu": nn.ReLU(), "gelu": nn.GELU()}[name.lower()]


def get_dropout(name: str, p: float):
    if p <= 0:
        return None
    return nn.AlphaDropout(p) if name.lower() == "selu" else nn.Dropout(p)


class FCN(nn.Module):
    """Direct fixed-window regressor. `hidden_dims` may be given explicitly (paper Table 3)
    or built from (num_expansion, scale_factor)."""

    def __init__(self, hidden_dims=None, num_expansion=2, scale_factor=1.0, activation="gelu",
                 use_bn=False, dropout=0.0, anchor=False, n_in=60, n_out=N_OUT, n_bands=1, extra_in=0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = build_hidden_dims(num_expansion, scale_factor)
        self.anchor = anchor
        self.n_out, self.n_bands = n_out, n_bands
        in_dim = n_in + (1 if anchor else 0) + extra_in
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(activation))
            do = get_dropout(activation, dropout)
            if do is not None:
                layers.append(do)
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out * n_bands))
        self.net = nn.Sequential(*layers)

    def forward(self, x, extra=None):  # x: (B, L, 1)
        x_flat = x.flatten(1)
        feats = [x_flat]
        if self.anchor:
            feats.append(x_flat[:, -1:])
        if extra is not None:
            feats.append(extra)
        out = self.net(torch.cat(feats, dim=1))
        if self.anchor:
            out = out + x_flat[:, -1:]
        return out.view(-1, self.n_out, self.n_bands)


class RecurrentNet(nn.Module):
    def __init__(self, kind="gru", hidden_dim=172, num_layers=2, dropout=0.15, n_out=N_OUT, n_bands=4, input_dim=1):
        super().__init__()
        cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[kind]
        kw = dict(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                  dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        if kind == "rnn":
            kw["nonlinearity"] = "tanh"
        self.rnn = cls(**kw)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_out * n_bands)
        self.n_out, self.n_bands = n_out, n_bands

    def forward(self, x, extra=None):
        if extra is not None:  # static features broadcast along time
            x = torch.cat([x, extra[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
        out, _ = self.rnn(x)
        h = self.drop(out[:, -1, :])
        return self.fc(h).view(-1, self.n_out, self.n_bands)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1), :])


class TransformerNet(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.0044, n_out=N_OUT, n_bands=4, max_len=60, input_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_out * n_bands)
        self.n_out, self.n_bands = n_out, n_bands

    def forward(self, x, extra=None):
        if extra is not None:
            x = torch.cat([x, extra[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
        x = self.pos_enc(self.input_proj(x))
        x = self.transformer(x).mean(dim=1)
        return self.fc(x).view(-1, self.n_out, self.n_bands)


# ---- Final hyper-parameters of the paper (Table 3 / Zenodo scripts) -------------------
PAPER_CONFIGS = {
    "fcn_w0": dict(kind="fcn", hidden_dims=[230, 922, 230], activation="gelu", use_bn=False, dropout=0.256843,
                   lr=4.84e-4, batch_size=16, anchor=True, bands=[0]),
    "fcn_w1": dict(kind="fcn", hidden_dims=[268, 1072, 268], activation="gelu", use_bn=False, dropout=0.0374,
                   lr=4.5e-4, batch_size=16, anchor=False, bands=[1]),
    "fcn_w2": dict(kind="fcn", hidden_dims=[110, 441, 110], activation="gelu", use_bn=False, dropout=0.0001,
                   lr=2.3e-4, batch_size=8, anchor=False, bands=[2]),
    "fcn_w3": dict(kind="fcn", hidden_dims=[238], activation="gelu", use_bn=True, dropout=0.1202,
                   lr=2.4e-3, batch_size=8, anchor=False, bands=[3]),
    "lstm": dict(kind="lstm", hidden_dim=208, num_layers=1, dropout=0.1245, lr=1.4e-3, batch_size=8, bands=[0, 1, 2, 3]),
    "gru": dict(kind="gru", hidden_dim=172, num_layers=2, dropout=0.1518, lr=5.8e-4, batch_size=8, bands=[0, 1, 2, 3]),
    "rnn": dict(kind="rnn", hidden_dim=233, num_layers=1, dropout=0.0298, lr=3.4e-4, batch_size=8, bands=[0, 1, 2, 3]),
    "transformer": dict(kind="transformer", d_model=128, nhead=8, num_layers=2, dropout=0.0044, lr=9.9e-4, batch_size=16,
                        bands=[0, 1, 2, 3]),
}


def build_model(cfg: dict, n_in: int = 60, extra_in: int = 0) -> nn.Module:
    kind = cfg["kind"]
    n_bands = len(cfg.get("bands", [0, 1, 2, 3]))
    if kind == "fcn":
        return FCN(hidden_dims=cfg.get("hidden_dims"), num_expansion=cfg.get("num_expansion", 2),
                   scale_factor=cfg.get("scale_factor", 1.0), activation=cfg.get("activation", "gelu"),
                   use_bn=cfg.get("use_bn", False), dropout=cfg.get("dropout", 0.0), anchor=cfg.get("anchor", False),
                   n_in=n_in, n_bands=n_bands, extra_in=extra_in)
    if kind in ("gru", "lstm", "rnn"):
        return RecurrentNet(kind=kind, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
                            dropout=cfg.get("dropout", 0.0), n_bands=n_bands, input_dim=1 + extra_in)
    if kind == "transformer":
        return TransformerNet(d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
                              dropout=cfg.get("dropout", 0.0), n_bands=n_bands, max_len=n_in, input_dim=1 + extra_in)
    from src.models.modern import build_modern  # lazy import (Phase 3 architectures)
    return build_modern(cfg, n_in=n_in, n_bands=n_bands, extra_in=extra_in)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
