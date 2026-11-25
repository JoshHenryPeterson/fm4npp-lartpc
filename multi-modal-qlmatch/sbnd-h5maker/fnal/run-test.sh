#!/bin/bash
# fm4npp #14 test10 on sbndbuild03 (SL7 image): one pair each, --max-events 10
export SBNDH5_SITE=/exp/sbnd/data/users/yuhw/production-prep/realdata-h5/site SBNDH5_MAKER=/exp/sbnd/data/users/yuhw/fm4npp-lartpc/multi-modal-qlmatch/sbnd-h5maker SBNDH5_STEP0=/exp/sbnd/app/users/yuhw/sbnd-wirecell-production-main/aurora/bin/step0-frameshift.sh
set +u; source /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/scripts/env-fnal.sh; set -u
run() { # sample id reco1
  cp "$3" /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/test10/in/ &&   /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/scripts/h5pair.sh /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/test10/in/$(basename "$3") /pnfs/sbnd/scratch/users/yuhw/prod-fermigrid/WCLocal2026Nov16/$1/$2/$2.tracking-pr.tar /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/test10/$4/sbnd_$4_test10.h5 10
}
mkdir -p /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/test10/data-bnb-opendata /exp/sbnd/data/users/yuhw/production-prep/realdata-h5/test10/data-offbeam-intime-run1
run opendata data_filtered_decoded_reco1-006b796b-db68-29f5-1f42-13ad421c1fec /pnfs/sbn/data_add/sbn_nd/poms_production/data/MCP2025C/v10_14_02/Fall25-Run1_BNB_Dev_bnblight/reco1/bnblight/00/data_filtered_decoded_reco1-006b796b-db68-29f5-1f42-13ad421c1fec.root data-bnb-opendata &
run beam-off data_filtered_decoded_reco1-0020ead9-98a5-99e9-e6a1-a03a66aa812b /pnfs/sbn/data_add/sbn_nd/poms_production/data/SBND2026A/v10_14_02_02/gen2_InTime-Run1/reco1/offbeamlight/00/data_filtered_decoded_reco1-0020ead9-98a5-99e9-e6a1-a03a66aa812b.root data-offbeam-intime-run1 &
wait
