#!/bin/bash

ARCH=$(uname -m)
BUILD_DIR=build-$ARCH

if [ ! -d $BUILD_DIR ]; then
  mkdir $BUILD_DIR
fi

pushd $BUILD_DIR
cmake-js compile --CMAKE_OSX_ARCHITECTURES=$ARCH --CMAKE_SOURCE_DIR=..
popd
