#!/bin/bash
set +e
set -x

export HOME="$WORKSPACE"
#cudf test
export LIBCUDF_KERNEL_CACHE_PATH="$WORKSPACE/.cache/rapids/cudf"
export PATH="/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/local/gcc7/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# FIXME: "source activate" line should not be needed
source /opt/conda/bin/activate rapids
env
nvidia-smi
conda list

TESTRESULTS_DIR="$WORKSPACE/testresults"
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

# run gtests
for gt in /rapids/cudf/cpp/build/gtests/*; do
   ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
   exitcode=$?
   if (( ${exitcode} != 0 )); then
      SUITEERROR=${exitcode}
      echo "FAILED: ${gt}"
   fi
done

# Python tests
export PYTHONPATH=\
/rapids/cudf/python/cudf:\
/rapids/cudf/python/dask_cudf:\
/rapids/cudf/python/custreamz:\
/rapids/cudf/python/nvstrings:\
${PYTHONPATH}

cd /rapids/cudf/python/cudf
py.test -n 6 --junitxml=${TESTRESULTS_DIR}/pytest-cudf.xml -v
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more python tests"
fi

cd /rapids/cudf/python/dask_cudf
py.test -n 6 --junitxml=${TESTRESULTS_DIR}/pytest-dask-cudf.xml -v
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more python tests"
fi

cd /rapids/cudf/python/custreamz
py.test -n 6 --junitxml=${TESTRESULTS_DIR}/pytest-custreamz.xml -v
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more python tests"
fi


#cugraph test
export LIBCUDF_KERNEL_CACHE_PATH="${WORKSPACE}/.jitcache"
export PATH="/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/local/gcc7/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# FIXME: "source activate" line should not be needed
source /opt/conda/bin/activate rapids

# Get datasets
cd /rapids/cugraph/datasets
bash ./get_test_data.sh
export RAPIDS_DATASET_ROOT_DIR=/rapids/cugraph/datasets

# Show environment
env
nvidia-smi
conda list

# Install pytest plugin for cugraph
conda install rapids-pytest-benchmark

TESTRESULTS_DIR="${WORKSPACE}/testresults"
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

# gtests
for gt in /rapids/cugraph/cpp/build/tests/*_TEST; do
   # FIXME: remove this ASAP
   ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
   exitcode=$?
   if (( ${exitcode} != 0 )); then
      SUITEERROR=${exitcode}
      echo "FAILED: ${gt}"
   fi
done

# Python tests
pytest --ignore=/rapids/cugraph/python/cugraph/raft --junitxml=${TESTRESULTS_DIR}/pytest.xml -v /rapids/cugraph/python
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more tests in /rapids/cugraph/python"
fi

#cuml test

export CUPY_CACHE_DIR="$WORKSPACE/tmp"
mkdir -p ${CUPY_CACHE_DIR}

# gtests
# FIXME: add /rapids/cuml/cpp/build/test/ml_mg when multi-gpus are available!
for gt in \
      /rapids/cuml/cpp/build/test/ml \
      /rapids/cuml/cpp/build/test/prims ; do
   ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
   exitcode=$?
   if (( ${exitcode} != 0 )); then
      SUITEERROR=${exitcode}
      echo "FAILED: ${gt}"
   fi
done

# Python tests
py.test --junitxml=${TESTRESULTS_DIR}/pytest.xml -v /rapids/cuml/python/cuml/test -m "not memleak"
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more python tests"
fi

#cuspatial test

TESTRESULTS_DIR="$WORKSPACE/testresults"
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

# Python tests
cd /rapids/cuspatial
py.test --junitxml=${TESTRESULTS_DIR}/pytest.xml -v python/cuspatial/cuspatial/tests
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more python tests"
fi
#daskcuda
TESTRESULTS_DIR="$WORKSPACE/testresults"
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

# Python tests
cd /rapids/dask-cuda/dask_cuda
py.test --junitxml=${TESTRESULTS_DIR}/pytest.xml -v 
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more tests in /rapids/dask-cuda/dask_cuda/tests"
fi

#Integration.sh test 

set -ex
export HOME="$WORKSPACE"
export LIBCUDF_KERNEL_CACHE_PATH="$WORKSPACE/.jitcache"
export PATH="/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/local/gcc7/bin:/usr/sbin:/usr/bin:/sbin:/bin"

. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Show env and current conda list"
env
nvidia-smi
conda list

export TESTRESULTS_DIR="$WORKSPACE/testresults"
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

gpuci_logger "Install conda packages needed by tests in rapids environment"
gpuci_conda_retry --condaretry_max_retries=10 install -y --freeze-installed requests

gpuci_logger "Run Python tests"
py.test --junitxml=${TESTRESULTS_DIR}/pytest.xml -v "$WORKSPACE/test"
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   gpuci_logger "FAILED: 1 or more tests in $WORKSPACE/test"
fi

#notebook.sh test 
set +e
set -x
set -o pipefail

export LIBCUDF_KERNEL_CACHE_PATH="$WORKSPACE/.jitcache"

source /opt/conda/bin/activate rapids

# PyTorch is intentionally excluded from our Docker images due
# to its size, but some notebooks still depend on it.
case "${CUDA_VER}" in
"10.1" | "10.2" | "11.0")
    conda install -y -c pytorch "pytorch=1.7"
    ;;
*)
    echo "Unsupported CUDA version for pytorch."
    echo "Not installing pytorch."
    ;;
esac


env
nvidia-smi
conda list

/test.sh 2>&1 | tee nbtest.log
EXITCODE=$?
python /rapids/utils/nbtestlog2junitxml.py nbtest.log


#rmm tests

# gtests
for gt in /rapids/rmm/build/gtests/*; do
   ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
   exitcode=$?
   if (( ${exitcode} != 0 )); then
      SUITEERROR=${exitcode}
      echo "FAILED: ${gt}"
   fi
done

# Python tests
cd /rapids/rmm/python
py.test --cache-clear --junitxml=${TESTRESULTS_DIR}/rmm_pytest.xml -v
exitcode=$?
if (( ${exitcode} != 0 )); then
   SUITEERROR=${exitcode}
   echo "FAILED: 1 or more tests in /rapids/rmm/python/tests"
fi

exit ${SUITEERROR}
