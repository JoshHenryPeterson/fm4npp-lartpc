#!/bin/bash
# Build uboone/hdf5maker (branch opensamples) against uboonecode v08_00_00_54 e17:prof.
# Run inside the SL7 container:  bash sl7.sh "bash $PWD/build.sh [TOP]"
TOP=${1:-/exp/uboone/app/users/yuhw/hdf5maker_build}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$TOP" && cd "$TOP" || exit 1
source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone_mcc9.sh
setup uboonecode v08_00_00_54 -q e17:prof
[ -d srcs ] || mrb newDev || exit 1
source localProducts_larsoft_*_e17_prof/setup
cd srcs
if [ ! -d hdf5maker ]; then
  git clone -b opensamples https://github.com/uboone/hdf5maker.git || exit 1
  # local patches (data mode etc.), applied once on a fresh clone
  for p in "$HERE"/patches/*.patch; do [ -e "$p" ] && (cd hdf5maker && git apply "$p") || true; done
fi
mrb updateDepsCM
mrbsetenv
mrb i -j4
