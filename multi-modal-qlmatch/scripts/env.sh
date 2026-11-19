# Source me: Polaris environment for qlfm
module use /soft/modulefiles >/dev/null 2>&1
module load conda/2025-09-25 >/dev/null 2>&1
export PY=/soft/applications/conda/2025-09-25/mconda3/bin/python
export QLFM_ROOT=${QLFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export QLFM_DATA=${QLFM_DATA:-/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data}
export QLFM_WORK=${QLFM_WORK:-/lus/eagle/projects/LArTPC_FM/jpeterso/fm4npp-lartpc-work}
export QLFM_REF_WORK=${QLFM_REF_WORK:-/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work}   # read-only reference work dir
export PIMM_ROOT=${PIMM_ROOT:-$QLFM_WORK/tools/pimm}
export BOX=${BOX:-$QLFM_WORK/jobs/box}
export QLFM_PBS_ACCOUNT=${QLFM_PBS_ACCOUNT:-LArTPC_FM}
export PYTHONPATH=$QLFM_ROOT:$PYTHONPATH
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TMPDIR=/tmp   # mpiexec sets a long TMPDIR -> DataLoader "AF_UNIX path too long" hang
