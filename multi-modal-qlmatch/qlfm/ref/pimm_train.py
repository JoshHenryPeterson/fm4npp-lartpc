"""Launch pimm training with our MicroBooNE dataset registered (run under torchrun in the pimm venv).

torchrun --nproc_per_node 4 qlfm/ref/pimm_train.py --config-file CFG --options save_path=OUT
Resumes automatically from the newest complete checkpoint in OUT/model (pimm: last/, last.prev/ or model_last.pth) (for chained 1-hour PBS jobs).
"""
import glob
import os
import sys

QLFM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if QLFM_ROOT not in sys.path:
    sys.path.insert(0, QLFM_ROOT)
from qlfm.paths import PIMM_ROOT  # noqa: E402

if PIMM_ROOT not in sys.path:
    sys.path.insert(0, PIMM_ROOT)

import pimm.datasets  # noqa: E402,F401
import qlfm.ref.pimm_ext  # noqa: E402,F401  (registers MicroBooNEPlaneDataset)
from pimm import train as pimm_train  # noqa: E402


def main():
    argv = sys.argv[1:]
    save = next((a.split("=", 1)[1] for a in argv if a.startswith("save_path=")), None)
    if save and os.path.isdir(os.path.join(save, "model")):
        from pimm.utils.path import latest_complete_checkpoint
        ck = latest_complete_checkpoint(os.path.join(save, "model"))
        if ck is not None:
            argv += ["resume=True", f"weight={ck}"]
            print(f"[pimm_train] resuming from {ck}", flush=True)
    sys.argv = [sys.argv[0]] + argv
    pimm_train.main()


if __name__ == "__main__":
    main()
