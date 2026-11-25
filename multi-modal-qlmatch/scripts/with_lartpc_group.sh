#!/bin/bash
# Run a command with the LArTPC_FM group (data/caches under /lus/eagle/projects/LArTPC_FM). Fresh logins and PBS jobs
# normally have it; otherwise use sg, re-exporting LD_LIBRARY_PATH, which the setgid sg binary strips (h5py needs the
# Cray libfabric path).
echo "groups: $(id -Gn)"
if id -Gn | tr ' ' '\n' | grep -qx LArTPC_FM; then
  exec "$@"
fi
exec sg LArTPC_FM -c "export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'; $(printf '%q ' "$@")"
