#!/usr/bin/env python
"""Round-2 downstream matrix (one task line per run, for run_queue.sh), identical protocol for every backbone
("R2P"): late-fused PMT light in the head; frozen at 0.1 % (2 seeds), 1 % and 10 % (1 seed); fine-tune at 0.1 %
and 1 % (1 seed, layer decay 0.9); a charge-only frozen diagnostic at 1 % (no late light). Step cap 3000,
test subsets of 10k events (T1: 3k). Reduced from the original plan because point-cloud / per-hit reference
backbones cost 10-30x more per step than qlfm cell tokens. Long runs first.

Usage: make_tasks_r2.py OUT_ROOT BACKBONE_KEY [BACKBONE_KEY ...] > tasks.txt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qlfm.paths import WORK as W  # noqa: E402

BACKBONES = {   # key: (prefix, --backbone, checkpoint)
    "qlfm_S_C": ("", "qlfm", f"{W}/pre/c_f0.90/fm.pt"),
    "qlfm_S_CL": ("", "qlfm", f"{W}/pre/cl_f0.90/fm.pt"),
    "qlfm_M_CL": ("", "qlfm", f"{W}/pre/m_cl_f0.90/fm.pt"),
    "qlfm_SD_CL": ("", "qlfm", f"{W}/pre/sd_cl/fm.pt"),
    "qlfm_SD_C": ("", "qlfm", f"{W}/pre/sd_c/fm.pt"),
    "fm4npp": ("@mamba ", "fm4npp", f"{W}/pre/fm4npp/fm.pt"),
    "panda": ("@pimm ", "panda", f"{W}/pre/panda"),
    "polarmae": ("@pimm ", "polarmae", f"{W}/pre/polarmae"),
}
TASKS = ("evt_nu", "flavor", "energy", "hit")


def runs():
    for lf, seeds in ((0.001, (0, 1)), (0.01, (0,)), (0.1, (0,))):
        for s in seeds:
            yield "frozen", lf, s, 1
    for lf in (0.001, 0.01):
        yield "finetune", lf, 0, 1
    yield "frozen", 0.01, 0, 0        # charge-only diagnostic (no late light)


def main():
    root, keys = sys.argv[1], sys.argv[2:]
    lines = []
    for key in keys:
        prefix, bb, ck = BACKBONES[key]
        for mode, lf, seed, late in runs():
            for t in TASKS:
                tag = f"{mode}{'' if late else '_nolight'}"
                extra = " --n-test 3000 --n-val 1000" if t == "hit" else " --n-test 10000 --n-val 2000"
                extra += " --max-steps 3000"
                if mode == "finetune":
                    extra += " --layer-decay 0.9"
                line = (f"{prefix}qlfm.train_downstream --task {t} --mode {mode} --backbone {bb} --fm {ck} "
                        f"--label-frac {lf:g} --seed {seed} --late-light {late}{extra} "
                        f"--out {root}/{key}/{t}/{tag}/f{lf:g}_s{seed}")
                long_run = lf >= 0.01
                lines.append((0 if long_run else 1, line))
    for _, l in sorted(lines, key=lambda x: x[0]):
        print(l)


if __name__ == "__main__":
    main()
