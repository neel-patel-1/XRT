#!/bin/bash

sudo apt-get update
echo y | sudo apt install asciidoc
echo y | sudo apt install libjson-c-dev
echo y | sudo apt install build-essential
echo y | sudo apt install cmake
echo y | sudo apt install uuid-dev
echo y | sudo apt install make
echo y | sudo apt install g++
echo y | sudo apt install git
echo y | sudo apt install libboost-all-dev


# idxd-config
ROOT=$(pwd)
cd $ROOT/third-party/idxd-config/
git submodule update --init .
./autogen.sh
./configure CFLAGS='-g -O2' --disable-logging --prefix=${PWD}/install --sysconfdir=/etc --libdir=${PWD}/install/lib64 --enable-test=yes
make -j
sudo make install -j

# ippcp
#sudo apt remove nasm
cd $ROOT/third-party/
wget https://www.nasm.us/pub/nasm/releasebuilds/2.16.02/nasm-2.16.02.tar.gz
tar -xvzf nasm-2.16.02.tar.gz
cd nasm-2.16.02
./configure
make -j
sudo make install -j

cd $ROOT/third-party/ipp-crypto
git submodule update --init --recursive .
ASM_NASM=/usr/bin/nasm cmake CMakeLists.txt -B_build -DARCH=intel64 -DCMAKE_BUILD_TYPE=Release
cd _build/
make -j $(( $(nproc) / 2 ))
