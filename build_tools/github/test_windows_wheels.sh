#!/bin/bash

set -e
set -x

PYTHON_VERSION=$1

docker container run \
    --rm scikit-learn/minimal-windows \
    powershell -Command "python -c 'import sflearn; sflearn.show_versions()'"

docker container run \
    -e SFLEARN_SKIP_NETWORK_TESTS=1 \
    -e OMP_NUM_THREADS=2 \
    -e OPENBLAS_NUM_THREADS=2 \
    --rm scikit-learn/minimal-windows \
    powershell -Command "pytest --pyargs sflearn"
