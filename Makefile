ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CXX=g++

all: main

kernels: obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a

obj/kernels/%.a: src/kernels/%.cpp
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/include

obj/%.o: src/%.cpp
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/include

obj/router.pb.o: src/protobuf/router.proto
	protoc --cpp_out=./src/protobuf/generated $<
	$(CXX) -std=c++17 -O3 -fPIC -c $(ROOT_DIR)/src/protobuf/generated/src/protobuf/router.pb.cc -o obj/router.pb.o -I$(ROOT_DIR)/src/protobuf/generated/

main: obj/main.o obj/iaa_offloads.o obj/dsa_offlaods.o obj/lzdatagen.o obj/pcg_basic.o obj/kernels/*.a obj/pointer_chase.o obj/router.pb.o obj/payload_gen.obj obj/thread_utils.o
	$(CXX) -std=c++17 -O3 -o $@ $^ -I$(ROOT_DIR)/include -L$(ROOT_DIR)/lib -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread