#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)
BACKEND="cpu"
GGUF_MODEL="<path-to-gguf-model>"

# refresh build folder
echo "[INFO] Refreshing build folder..."
cd ${PROJECT_DIR}
BUILD_FOLDER="build_${BACKEND}"
if [ -d ${BUILD_FOLDER} ]; then
    rm -rf ${BUILD_FOLDER}
fi
mkdir ${BUILD_FOLDER}


# configure project
cd ${PROJECT_DIR}/${BUILD_FOLDER}
echo "[INFO] Configuring project..."
if [ ${BACKEND} == "cpu" ]; then
    cmake .. \
        -DGGML_CPU=ON
elif [ ${BACKEND} == "cuda" ]; then
    cmake .. \
        -DGGML_CUDA=ON
fi

# build project
echo "[INFO] Building project..."
make -j16


echo "[INFO] All done."