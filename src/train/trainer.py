"""Training loop replicating the Zenodo final-training protocol of Kim et al. (2026).

Adam, MSE on log10 irradiance, ReduceLROnPlateau(factor 0.5, patience 15), grad-clip 1.0,
max 300 epochs, early stopping on validation loss (patience 30), best-validation checkpoint.
All of these are configurable (Phase-2/3 experiments use shorter budgets where stated).
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.models.nets import build_model

_DEVICE = None
MIN_FREE = 2 << 30          # a device needs >= 2 GB free to be used at all
ROOMY_FREE = 4 << 30        # workers are spread (round-robin by pid) over the devices with > 4 GB free


def _smi_table():
    """{uuid: (free_bytes, util_percent)} from nvidia-smi, or None if it is not available.
    Used to look at the GPUs WITHOUT creating a CUDA context on each of them (torch.cuda.mem_get_info would
    leave ~400 MB of this process on every visible GPU, including the full ones)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return None
    tab = {}
    for line in out.splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) < 4 or not p[0].startswith("GPU-"):
            continue
        try:
            used, total, util = float(p[1]), float(p[2]), float(p[3])
        except ValueError:          # '[Unknown Error]' fields of a broken GPU
            continue
        tab[p[0].lower()] = ((total - used) * 2 ** 20, util)
    return tab or None


def _cuda_candidates():
    """([(free_bytes, cuda_index)] of usable devices, first error text or None).
    Free memory / utilisation come from nvidia-smi (matched by UUID, so CUDA_VISIBLE_DEVICES and a broken,
    non-enumerated GPU do not shift the indices); torch.cuda.mem_get_info is the fallback."""
    cands, err, tab = [], None, _smi_table()
    for i in range(torch.cuda.device_count()):
        try:
            uuid = getattr(torch.cuda.get_device_properties(i), "uuid", None)
            key = "gpu-" + str(uuid).lower().replace("gpu-", "") if uuid is not None else None
            if tab is not None and key in tab:
                free, util = tab[key]
            else:
                free, _ = torch.cuda.mem_get_info(i)
                try:
                    util = float(torch.cuda.utilization(i))
                except Exception:  # noqa: BLE001 - NVML unavailable
                    util = 0.0
            if util < 90 and free > MIN_FREE:
                cands.append((free, i))
        except Exception as e:  # noqa: BLE001 - a broken device, or CUDA unusable in this process
            err = err or f"cuda:{i}: {str(e).splitlines()[0] if str(e) else type(e).__name__}"
    return cands, err


def get_device() -> torch.device:
    """Training device. FISM_DEVICE = auto (default) | cpu | cuda | cuda:N  (chosen once per process, printed).
    auto: among the CUDA devices with >= 2 GB free and < 90 % utilisation, workers are spread over those with
    > 4 GB free (round-robin by pid), else the one with the most free memory.
    FISM_REQUIRE_GPU=1: never fall back to the CPU silently - if every GPU is full, wait up to FISM_GPU_WAIT_S
    (default 900 s) for one to free up, then raise; a CUDA error (e.g. a poisoned fork) raises at once."""
    import os
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    want = os.environ.get("FISM_DEVICE", "auto").lower()
    require = os.environ.get("FISM_REQUIRE_GPU", "0") == "1"
    quiet = os.environ.get("FISM_DEVICE_QUIET", "0") == "1"
    dev, note = torch.device("cpu"), ""
    if want.startswith("cuda"):
        if torch.cuda.is_available():
            dev = torch.device(want)
        elif require:
            raise RuntimeError(f"FISM_DEVICE={want} but torch.cuda.is_available() is False (FISM_REQUIRE_GPU=1)")
        else:
            note = " (CUDA not available)"
    elif want == "auto":
        if not torch.cuda.is_available():
            if require:
                raise RuntimeError("FISM_REQUIRE_GPU=1 but torch.cuda.is_available() is False "
                                   "(CPU-only torch build, no driver, or CUDA_VISIBLE_DEVICES hides every GPU)")
            note = " (CUDA not available)"
        else:
            cands, err = _cuda_candidates()
            wait, t0 = float(os.environ.get("FISM_GPU_WAIT_S", "900")), time.time()
            while not cands and err is None and require and time.time() - t0 < wait:   # every GPU is full: wait
                if not quiet:
                    print(f"[trainer] no GPU with >= 2 GB free - waiting ({time.time() - t0:.0f}/{wait:.0f} s)", flush=True)
                time.sleep(30)
                cands, err = _cuda_candidates()
            if cands:
                cands.sort(reverse=True)
                roomy = [c for c in cands if c[0] > ROOMY_FREE] or cands[:1]
                dev = torch.device(f"cuda:{roomy[os.getpid() % len(roomy)][1]}")
            else:
                reason = err or "no GPU with >= 2 GB free and < 90 % utilisation"
                if require:
                    raise RuntimeError(f"FISM_REQUIRE_GPU=1 and no usable CUDA device: {reason} "
                                       "(FISM_REQUIRE_GPU=0 allows the CPU)")
                note = f" (CUDA available but unusable here: {reason})"
    _DEVICE = dev
    if not quiet:
        name = f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""
        print(f"[trainer] device: {dev}{name}{note}", flush=True)
    return dev


def reselect_device():
    """Forget the cached choice and pick again (used after a CUDA out-of-memory error: another GPU may be free)."""
    global _DEVICE
    _DEVICE = None
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return get_device()


def force_cpu():
    """Switch this process to the CPU (used after a CUDA out-of-memory error)."""
    global _DEVICE
    _DEVICE = torch.device("cpu")
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


OOM = getattr(torch, "OutOfMemoryError", getattr(torch.cuda, "OutOfMemoryError", RuntimeError))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensors(xl, yl, bands):
    x = torch.tensor(xl, dtype=torch.float32).unsqueeze(-1)          # (N, L, 1)
    y = torch.tensor(yl[:, :, bands], dtype=torch.float32)           # (N, 180, nb)
    return x, y


def train_model(cfg: dict, Xtr_log, Ytr_log, Xva_log, Yva_log, seed: int = 0, epochs: int = 300,
                patience: int = 30, sched_patience: int = 15, loss_weights=None, extra_tr=None,
                extra_va=None, verbose: bool = False, max_minutes: float | None = None, augment_fn=None):
    """Returns (model, history). Inputs are log10 arrays: X (N,L), Y (N,180,4)."""
    set_seed(seed)
    dev = get_device()
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    bands = cfg.get("bands", [0, 1, 2, 3])
    n_in = Xtr_log.shape[1]
    extra_in = 0 if extra_tr is None else extra_tr.shape[1]
    model = build_model(cfg, n_in=n_in, extra_in=extra_in).to(dev)
    xtr, ytr = _tensors(Xtr_log, Ytr_log, bands)
    xva, yva = _tensors(Xva_log, Yva_log, bands)
    xtr, ytr, xva, yva = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)
    etr = None if extra_tr is None else torch.tensor(extra_tr, dtype=torch.float32, device=dev)
    eva = None if extra_va is None else torch.tensor(extra_va, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=sched_patience)
    bs = int(cfg["batch_size"])
    w = None if loss_weights is None else torch.tensor(loss_weights, dtype=torch.float32, device=dev)

    def loss_fn(pred, y, idx=None):
        se = (pred - y) ** 2
        if w is not None and idx is not None:
            return (se.mean(dim=(1, 2)) * w[idx]).sum() / w[idx].sum()
        return se.mean()

    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        tot = 0.0
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and cfg.get("use_bn", False):
                continue
            opt.zero_grad()
            xb, yb = xtr[idx], ytr[idx]
            if augment_fn is not None:
                xb, yb = augment_fn(xb, yb, g)             # augment_fn draws on the CPU generator
            pred = model(xb, None if etr is None else etr[idx])
            loss = loss_fn(pred, yb, idx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = ((model(xva, eva) - yva) ** 2).mean().item()
        sched.step(vl)
        hist.append((ep, tot / len(xtr), vl, opt.param_groups[0]["lr"]))
        if vl < best - 1e-9:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and ep % 10 == 0:
            print(f"ep {ep:3d} train {tot / len(xtr):.5f} val {vl:.5f} lr {opt.param_groups[0]['lr']:.2e}")
        if bad >= patience:
            break
        if max_minutes is not None and (time.time() - t0) / 60 > max_minutes:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_val": best, "epochs_run": len(hist), "seconds": time.time() - t0}


@torch.no_grad()
def predict_log(model, X_log, extra=None, bands_out=None):
    dev = next(model.parameters()).device
    x = torch.tensor(X_log, dtype=torch.float32, device=dev).unsqueeze(-1)
    e = None if extra is None else torch.tensor(extra, dtype=torch.float32, device=dev)
    return model(x, e).cpu().numpy()
