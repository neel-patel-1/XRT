ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CXX=g++

all: main

kernels: obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a

obj/kernels/%.a: src/kernels/%.cpp
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/include

main: obj/main.o obj/