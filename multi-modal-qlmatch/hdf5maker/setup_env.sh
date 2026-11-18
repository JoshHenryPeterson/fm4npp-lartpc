# Source inside the SL7 container to get lar + the locally built hdf5maker.
#   source setup_env.sh [TOP]
_TOP=${1:-/exp/uboone/app/users/yuhw/hdf5maker_build}
source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone_mcc9.sh
setup uboonecode v08_00_00_54 -q e17:prof
source $_TOP/localProducts_larsoft_*_e17_prof/setup
setup hdf5maker v09_25_00 -q e17:prof
_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FHICL_FILE_PATH=$_HERE/fcl:$FHICL_FILE_PATH
which lar
