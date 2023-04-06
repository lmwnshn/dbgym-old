#!/bin/bash

set -euxo pipefail

sudo chown -R dbgym:dbgym /dbgym
cd /app
# TODO(WAN): hack around autogluon bug, see also https://github.com/autogluon/autogluon/issues/1020
#dbgym-dbgym    | OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
#dbgym-dbgym    |        -468594.7268     = Validation score   (-root_mean_squared_error)
#dbgym-dbgym    |        1.49s    = Training   runtime
#dbgym-dbgym    |        0.51s    = Validation runtime
#dbgym-dbgym    | Fitting model: KNeighborsDist ... Training model for up to 288.58s of the 288.58s of remaining time.
#dbgym-dbgym    | OpenBLAS : Program is Terminated. Because you tried to allocate too many memory regions.
#dbgym-dbgym    | This library was built to support a maximum of 128 threads - either rebuild OpenBLAS
#dbgym-dbgym    | with a larger NUM_THREADS value or set the environment variable OPENBLAS_NUM_THREADS to
#dbgym-dbgym    | a sufficiently small number. This error typically occurs when the software that relies on
#dbgym-dbgym    | OpenBLAS calls BLAS functions from many threads in parallel, or when your computer has more
#dbgym-dbgym    | cpu cores than what OpenBLAS was configured to handle.
#dbgym-dbgym    | /docker-entrypoint.sh: line 7:     9 Segmentation fault      (core dumped) python3 -u -m dbgym
export OPENBLAS_NUM_THREADS=32
export GOTO_NUM_THREADS=32
export OMP_NUM_THREADS=32
python3 -u -m dbgym
