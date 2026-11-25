"""GOES XRS -> FISM2 0.1-0.8 nm input converter (paper-1, P2).

The benchmark models are trained on the FISM2 0.1-0.8 nm band, which is not available in real time; the
real-time observable is GOES XRS-B (0.1-0.8 nm) and XRS-A (0.05-0.4 nm). The converter maps a 60-minute
GOES window to the FISM2 window so that FISM2-trained models can be driven by GOES.

Two variants (all in log10 space):
  linear : ridge regression per minute sample on [1, logB, logA, logB-logA, dlogB, dlogA]; minutes without
           XRS-A use a B-only model. Fitted on the training events only.
  fcn    : a small fully-connected network on the whole 60x2 window -> 60 outputs (residual on logB),
           early-stopped on the validation events.

The FCN is fitted in a separate 'spawn' process (see FCNConverter.fit) and its forward pass in the parent
process is evaluated in pure NumPy/SciPy without importing or initializing PyTorch in the parent process.
This guarantees that forked training workers never encounter a poisoned CUDA or Autograd state.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


def _feats(lb: np.ndarray, la: np.ndarray | None) -> np.ndarray:
    """lb, la: (N, L) log10 series -> (N*L, F) feature rows (per-minute)."""
    dlb = np.diff(lb, axis=1, prepend=lb[:, :1])
    cols = [np.ones_like(lb), lb, dlb]
    if la is not None:
        dla = np.diff(la, axis=1, prepend=la[:, :1])
        cols += [la, lb - la, dla]
    return np.stack(cols, axis=-1).reshape(-1, len(cols))


class LinearConverter:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.w_ab = None
        self.w_b = None

    def _solve(self, F, y):
        A = F.T @ F + self.ridge * np.eye(F.shape[1])
        return np.linalg.solve(A, F.T @ y)

    def fit(self, Xb, Xa, Xf):
        lb, lf = np.log10(Xb), np.log10(Xf)
        ok_a = np.all(np.isfinite(Xa) & (Xa > 0), axis=1)
        self.w_b = self._solve(_feats(lb, None), lf.ravel())
        if ok_a.sum() >= 20:
            la = np.log10(Xa[ok_a])
            self.w_ab = self._solve(_feats(lb[ok_a], la), lf[ok_a].ravel())
        return self

    def predict_log(self, Xb, Xa):
        lb = np.log10(Xb)
        out = (_feats(lb, None) @ self.w_b).reshape(lb.shape)
        if self.w_ab is not None:
            ok_a = np.all(np.isfinite(Xa) & (Xa > 0), axis=1)
            if ok_a.any():
                la = np.log10(Xa[ok_a])
                out[ok_a] = (_feats(lb[ok_a], la) @ self.w_ab).reshape(lb[ok_a].shape)
        return out


def _fit_fcn_worker(args):
    """Runs in the spawned child: fit and return (mu, sd, state_dict as numpy, info)."""
    hidden, epochs, patience, lr, seed, Xb, Xa, Xf, Xb_va, Xa_va, Xf_va = args
    c = FCNConverter(hidden=hidden, epochs=epochs, patience=patience, lr=lr, seed=seed)
    c._fit_inproc(Xb, Xa, Xf, Xb_va, Xa_va, Xf_va)
    return c.mu, c.sd, {k: v.detach().cpu().numpy() for k, v in c.model.state_dict().items()}, c.fit_info


class FCNConverter:
    """log F = log B + MLP([logB, logA or logB]) on the whole window (residual learning)."""

    def __init__(self, hidden: int = 128, epochs: int = 300, patience: int = 30, lr: float = 1e-3, seed: int = 0):
        self.hidden, self.epochs, self.patience, self.lr, self.seed = hidden, epochs, patience, lr, seed
        self._model = None
        self._state_np = None
        self._L = None
        self.mu = self.sd = None
        self.fit_info = {}

    @property
    def model(self):
        if self._model is None and self._state_np is not None and self._L is not None:
            import torch
            net = self._build(self._L)
            net.load_state_dict({k: torch.tensor(v) for k, v in self._state_np.items()})
            net.eval()
            self._model = net
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    def _inputs(self, Xb, Xa):
        lb = np.log10(Xb)
        la = np.where(np.all(np.isfinite(Xa) & (Xa > 0), axis=1)[:, None], np.log10(np.where(Xa > 0, Xa, 1.0)), lb)
        return np.concatenate([lb, la], axis=1).astype(np.float32)

    def _build(self, L: int):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(2 * L, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.hidden), nn.GELU(),
                             nn.Linear(self.hidden, L))

    def fit(self, Xb, Xa, Xf, Xb_va=None, Xa_va=None, Xf_va=None):
        """Default: fit in a fresh 'spawn' process and keep weights as NumPy arrays so this (parent) process
        never runs autograd or even imports torch before training workers are forked."""
        self._L = int(Xb.shape[1])
        inproc = os.environ.get("FISM_CONV_SUBPROCESS", "1") != "1" or mp.parent_process() is not None
        if inproc:
            return self._fit_inproc(Xb, Xa, Xf, Xb_va, Xa_va, Xf_va)
        args = (self.hidden, self.epochs, self.patience, self.lr, self.seed, Xb, Xa, Xf, Xb_va, Xa_va, Xf_va)
        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as p:
            self.mu, self.sd, state, self.fit_info = p.apply(_fit_fcn_worker, (args,))
        self._state_np = state
        self._model = None
        return self

    def _fit_inproc(self, Xb, Xa, Xf, Xb_va=None, Xa_va=None, Xf_va=None):
        import torch
        from src.train.trainer import set_seed
        self._L = int(Xb.shape[1])
        set_seed(self.seed)
        dev = torch.device("cpu")      # never touch CUDA here (see fit)
        Z = self._inputs(Xb, Xa)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        net = self._build(Xb.shape[1]).to(dev)
        x = torch.tensor((Z - self.mu) / self.sd, device=dev)
        y = torch.tensor(np.log10(Xf) - np.log10(Xb), dtype=torch.float32, device=dev)
        has_va = Xb_va is not None and len(Xb_va) > 1
        if has_va:
            xv = torch.tensor((self._inputs(Xb_va, Xa_va) - self.mu) / self.sd, device=dev)
            yv = torch.tensor(np.log10(Xf_va) - np.log10(Xb_va), dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, ep = np.inf, None, 0, 0
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(x), generator=g).to(dev)
            for i in range(0, len(x), 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = ((net(x[idx]) - y[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = ((net(xv) - yv) ** 2).mean().item() if has_va else ((net(x) - y) ** 2).mean().item()
            if vl < best - 1e-9:
                best, best_state, bad = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self._model = net
        self._state_np = {k: v.detach().cpu().numpy() for k, v in best_state.items()}
        self.fit_info = {"best_val_mse": float(best), "epochs_run": ep + 1}
        return self

    def predict_log(self, Xb, Xa):
        Z = ((self._inputs(Xb, Xa) - self.mu) / self.sd).astype(np.float32)
        if self._state_np is not None:
            from scipy.special import erf
            s = self._state_np
            inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
            h1 = Z @ s["0.weight"].T + s["0.bias"]
            h1 = np.float32(0.5) * h1 * (np.float32(1.0) + erf(h1 * inv_sqrt2).astype(np.float32))
            h2 = h1 @ s["2.weight"].T + s["2.bias"]
            h2 = np.float32(0.5) * h2 * (np.float32(1.0) + erf(h2 * inv_sqrt2).astype(np.float32))
            r = h2 @ s["4.weight"].T + s["4.bias"]
            return np.log10(Xb) + r
        import torch
        dev = next(self.model.parameters()).device
        Zt = torch.tensor(Z, device=dev)
        with torch.no_grad():
            r = self.model(Zt).cpu().numpy()
        return np.log10(Xb) + r


def make_converter(kind: str, seed: int = 0):
    if kind in ("linear", "goes_conv"):
        return LinearConverter()
    if kind in ("fcn", "goes_convnn"):
        return FCNConverter(seed=seed)
    raise ValueError(kind)
