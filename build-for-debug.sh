#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)
BACKEND="cpu"
BUILD_FOLDER="build_${BACKEND}_profiler"


# refresh build folder
echo "[INFO] Refreshing build folder..."
cd ${PROJECT_DIR}
if [ -d ${BUILD_FOLDER} ]; then
    rm -rf ${BUILD_FOLDER}
fi
mkdir ${BUILD_FOLDER}


# configure project
cd ${PROJECT_DIR}/${BUILD_FOLDER}
echo "[INFO] Configuring project..."
if [ ${BACKEND} == "cpu" ]; then
    cmake .. \
        -DGGML_CPU=ON \
        -DLLAMA_GRAPH_PROFILER=ON
elif [ ${BACKEND} == "cuda" ]; then
    cmake .. \
        -DGGML_CUDA=ON \
        -DLLAMA_GRAPH_PROFILER=ON \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-10 \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-10 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-10
fi

# build project
echo "[INFO] Building project..."
make -j16


echo "[INFO] All done."



echo ">> [INFO] All finished."
