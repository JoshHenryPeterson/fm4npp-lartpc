# Source INSIDE the SL7 container (Aurora: sbnd-wirecell-production/aurora/env/aurora/in-aurora-sl7.sh,
# SL7_SETUP=<this file>).  sbndcode v10_14_02_03 (the Gen2 reco1 MC version) + gallery for PyROOT,
# plus h5py / uproot for the UPS python 3.9 (wheels unpacked into $SBNDH5_SITE, see README).
Y=${Y:-/lus/flare/projects/neutrinoGPU/yuhw}
UPS_TREE=${UPS_TREE:-/lus/flare/projects/neutrinoGPU/scisoft/larsoft}
source $UPS_TREE/setup >/dev/null 2>&1
export PRODUCTS=$Y/products:$PRODUCTS
setup sbndcode v10_14_02_03 -q e26:prof
setup gallery v1_22_06 -q e26:prof
export SBNDH5_SITE=${SBNDH5_SITE:-$Y/tools/sl7-py39-h5/site}
export PYTHONPATH=$SBNDH5_SITE${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
