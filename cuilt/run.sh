#!/bin/bash
sysOS=`uname -s`
if [ $sysOS == "Darwin" ];then
	echo "using MacOS"
    CMAKE_SCRIPT=cmake
elif [ $sysOS == "Linux" ];then
	echo "using Linux"
    CMAKE_SCRIPT=cmake3
else
	echo "Other OS: $sysOS"
    CMAKE_SCRIPT=cmake
fi

# cd lithosim/pvbandsim_gpu
# mkdir -p build
# cd build
# $CMAKE_SCRIPT ..
# make

# cd ../../../
mkdir -p build
cd build
$CMAKE_SCRIPT ..
make
ln -s ../evaluation/Kernels Kernels

./cuilt -input ../Benchmarks/M1_test1.glp -output M1_test_mask.glp
