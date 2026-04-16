#!/bin/bash

# set variables
CURRENT_DIR=$(dirname "$0")
PROJECT_DIR=$(dirname "$CURRENT_DIR")
BUILD_DIR="build-with-cuda"
CMAKE_ARGS="-DGGML_CUDA=ON"


# check if sudoers
if [ "$EUID" -ne 0 ]; then
    SUDO_ER='sudo'
else
    SUDO_ER=''


# refresh build directory
cd ${PROJECT_DIR}
echo ">> [INFO] Refreshing '${BUILD_DIR}' directory ... "
if [ -d ${BUILD_DIR} ]; then
    rm -rf ${BUILD_DIR}
fi

# install dependencies
echo ">> [INFO] Installing dependencies ... "
${SUDO_ER} apt install libcurl4-openssl-dev

# build project
echo ">> [INFO] Building project  ... "
cd ${PROJECT_DIR}
cmake -B ${BUILD_DIR} ${CMAKE_ARGS}
cd ${BUILD_DIR} && make -j96

echo ">> [INFO] All finished."

