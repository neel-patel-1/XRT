#!/bin/bash

sudo apt-get update
echo y | sudo apt install build-essential
echo y | sudo apt install cmake
echo y | sudo apt install uuid-dev
echo y | sudo apt install make
echo y | sudo apt install nasm
echo y | sudo apt install g++
echo y | sudo apt install git

# idxd-config
ROOT=$(pwd)
cd $ROOT/third-party/idxd-config/
git submodule update --init .
./autogen.sh
./configure CFLAGS='-g -O2' --disable-logging --prefix=${PWD}/install --sysconfdir=/etc --libdir=${PWD}/install/lib64 --enable-test=yes
make -j
sudo make install -j