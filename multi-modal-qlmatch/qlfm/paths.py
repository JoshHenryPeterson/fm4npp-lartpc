"""Work-directory locations (dependency-free; safe to import from anywhere, including pimm configs' helpers).

WORK      writable work dir (cache/, pre/, paper/, new runs); env QLFM_WORK
REF_WORK  read-only reference work dir holding published run results (r2p/, r3/, r4/, round2/); env QLFM_REF_WORK
PIMM_ROOT pimm checkout; env PIMM_ROOT, default $WORK/tools/pimm
"""
import os

_YUHW = "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work"
WORK = os.environ.get("QLFM_WORK", _YUHW)
REF_WORK = os.environ.get("QLFM_REF_WORK", _YUHW)
PIMM_ROOT = os.environ.get("PIMM_ROOT", os.path.join(WORK, "tools/pimm"))
