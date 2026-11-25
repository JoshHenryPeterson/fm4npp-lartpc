# fm4npp #14: FNAL (FermiGrid / sbndbuild03) twin of sbnd-h5maker/env.sh. Source INSIDE the SL7 image.
# sbndcode v10_14_02_03 (also gives run_frameshift.fcl) + gallery from CVMFS; the wheels in $SBNDH5_SITE.
source /cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh > /dev/null 2>&1
export PRODUCTS=$PRODUCTS:/cvmfs/larsoft.opensciencegrid.org/products
setup sbndcode v10_14_02_03 -q e26:prof
setup gallery v1_22_06 -q e26:prof
setup ifdhc 2>/dev/null
export SBNDH5_SITE=${SBNDH5_SITE:?set SBNDH5_SITE to the unpacked wheels}
export PYTHONPATH=$SBNDH5_SITE${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
