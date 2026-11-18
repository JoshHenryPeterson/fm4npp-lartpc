#!/bin/bash
# Run a command inside the SL7 dev container (uboonecode v08 is built for SL7).
#   bash sl7.sh 'source setup_env.sh && lar ...'      (no args: interactive shell)
IMG=/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest
APPT=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer
BINDS=/cvmfs,/exp,/nashome,/pnfs/uboone,/run/user,/etc/hostname,/etc/hosts,/etc/krb5.conf
if [ $# -eq 0 ]; then exec $APPT shell --shell=/bin/bash -B $BINDS $IMG; fi
exec $APPT exec -B $BINDS $IMG /bin/bash -c "$*"
