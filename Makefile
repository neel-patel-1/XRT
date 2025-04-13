ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CXX=g++
INCLUDES=-I$(ROOT_DIR)/third-party/idxd-config/test/algorithms -I$(ROOT_DIR)/third-party/idxd-config/test -I$(ROOT_DIR)/include -I$(ROOT_DIR)/third-party/idxd-config/accfg -I$(ROOT_DIR)/third-party/idxd-config -I$(ROOT_DIR) -I$(ROOT_DIR)/third-party/ipp-crypto/_build/RELEASE/include -I$(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/include -I$(ROOT_DIR)/include/inline

all: main_xrt

kernels: obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a

obj/kernels/%.a: src/kernels/%.cpp
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/include

obj/%.o: src/%.cpp
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ $(INCLUDES) -fpermissive

obj/router.pb.o: src/protobuf/generated/src/protobuf/router.pb.cc
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/src/protobuf/generated/

src/protobuf/generated/src/protobuf/router.pb.h: src/protobuf/router.proto
	protoc --cpp_out=./src/protobuf/generated $<

src/protobuf/generated/src/protobuf/router.pb.cc: src/protobuf/router.proto
	protoc --cpp_out=./src/protobuf/generated $<

main_xrt: src/main.cpp obj/iaa_offloads.o obj/dsa_offloads.o obj/lzdatagen.o obj/pcg_basic.o  obj/pointer_chase.o src/protobuf/generated/src/protobuf/router.pb.h obj/router.pb.o obj/payload_gen.o obj/thread_utils.o obj/numa_mem.o obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a
	$(CXX) -std=c++17 -O3 -o $@ $^ -DLATENCY -DTHROUGHPUT -DPERF -DPOISSON -DJSQ -DLOST_ENQ_TIME $(INCLUDES) -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config -lz -lcrypto -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a -fpermissive

clean:
	rm -f obj/*.o obj/kernels/*.a obj/router.pb.o main_xrt
	rm -rf src/protobuf/generated/*