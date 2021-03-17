#!/bin/bash

BUILDTYPE='Release'
#BUILDTYPE='Debug'

rm -rf -- ../build/${BUILDTYPE} && \
mkdir -p ../build/${BUILDTYPE} && \
cd ../build/${BUILDTYPE} && \

if [ "${BUILDTYPE}" == "Debug" ]; then
	cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug ../../source/
else
	cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release ../../source/
fi

make
