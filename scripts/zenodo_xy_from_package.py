"""Build data/processed/zenodo_test_xy.npz and data/processed/zenodo_full.npz from the authors' Zenodo package.

    python scripts/zenodo_xy_from_package.py external/zenodo/raw/2_evaluation/3_evaluation/all_model_predictions_same_ytrue.npz

Source: Kim et al. (2026), Zenodo doi:10.5281/zenodo.21400071 (CC BY 4.0), file 3_evaluation/all_model_predictions_same_ytrue.npz
inside 2_evaluation.zip. zenodo_test_xy.npz holds X_real (96x60), Y_true (96x180x4) and event_names (the 96 test flares);
zenodo_full.npz adds the authors' prediction arrays Y_<model>. The runs of the paper used the same arrays decoded from a
checksummed transcript (scripts/decode_zenodo_xy.py), which differs from the package by at most 0.0115 % (docs/data_card.md).

Safety: the numeric arrays are read with allow_pickle=False. The package stores event_names as a pickled object array; it is read
only if the file has the SHA-256 of the published package, and then with an unpickler that accepts numpy array reconstruction only.
"""
import hashlib
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHA256 = "114c16193c35e23b641b79459f9ee598df8d91a29db495d05d94b8b375a4e1ba"     # all_model_predictions_same_ytrue.npz, Zenodo v2


class _NumpyOnly(pickle.Unpickler):
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def event_names(src: Path) -> np.ndarray:
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != SHA256:
        sys.exit(f"{src}: SHA-256 {digest} is not that of the published package ({SHA256}); not reading its pickled event names")
    with zipfile.ZipFile(src) as zf:
        f = io.BytesIO(zf.read("event_names.npy"))
    version = np.lib.format.read_magic(f)
    shape, _, dtype = np.lib.format._read_array_header(f, version)
    if dtype != np.dtype(object):
        return np.lib.format.read_array(io.BytesIO(f.getvalue()), allow_pickle=False).astype(str)
    arr = _NumpyOnly(f).load()
    return np.array([str(e) for e in np.asarray(arr).reshape(shape)])


def main(src):
    src = Path(src)
    names = event_names(src)
    z = np.load(src, allow_pickle=False)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "zenodo_test_xy.npz", X_real=z["X_real"].astype(np.float64), Y_true=z["Y_true"].astype(np.float64), event_names=names)
    full = {k: z[k] for k in z.files if k != "event_names"}
    np.savez(out / "zenodo_full.npz", event_names=names, **full)
    print(f"wrote {out / 'zenodo_test_xy.npz'} and {out / 'zenodo_full.npz'}: {len(names)} events, models "
          f"{sorted(k[2:] for k in z.files if k.startswith('Y_') and k != 'Y_true')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
