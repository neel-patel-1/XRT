ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CXX=g++
INCLUDES=-I$(ROOT_DIR)/third-party/idxd-config/test/algorithms -I$(ROOT_DIR)/third-party/idxd-config/test -I$(ROOT_DIR)/include -I$(ROOT_DIR)/third-party/idxd-config/accfg -I$(ROOT_DIR)/third-party/idxd-config -I$(ROOT_DIR) -I$(ROOT_DIR)/third-party/ipp-crypto/_build/RELEASE/include -I$(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/include -I$(ROOT_DIR)/include/inline

all: mmp ufh ddh mg dg dmd

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

mmp: src/mmp.cpp obj/racer_opts.o obj/iaa_offloads.o obj/dsa_offloads.o obj/lzdatagen.o obj/pcg_basic.o  obj/pointer_chase.o src/protobuf/generated/src/protobuf/router.pb.h obj/router.pb.o obj/payload_gen.o obj/thread_utils.o obj/numa_mem.o obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a obj/stats.o
	$(CXX) -std=c++17 -march=native -O3 -o $@ $^ -DEXETIME $(INCLUDES) -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config -lz -lcrypto -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a -fpermissive
ufh: src/ufh.cpp obj/racer_opts.o obj/iaa_offloads.o obj/dsa_offloads.o obj/lzdatagen.o obj/pcg_basic.o  obj/pointer_chase.o src/protobuf/generated/src/protobuf/router.pb.h obj/router.pb.o obj/payload_gen.o obj/thread_utils.o obj/numa_mem.o obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a obj/stats.o
	$(CXX) -std=c++17 -march=native -O3 -o $@ $^ -DEXETIME $(INCLUDES) -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config -lz -lcrypto -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a -fpermissive
ddh: src/ddh.cpp obj/racer_opts.o obj/iaa_offloads.o obj/dsa_offloads.o obj/lzdatagen.o obj/pcg_basic.o  obj/pointer_chase.o src/protobuf/generated/src/protobuf/router.pb.h obj/router.pb.o obj/payload_gen.o obj/thread_utils.o obj/numa_mem.o obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a obj/stats.o
	$(CXX) -std=c++17 -march=native -O3 -o $@ $^ -DEXETIME $(INCLUDES) -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config -lz -lcrypto -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a -fpermissive
mg: src/mg.cpp obj/racer_opts.o obj/iaa_offloads.o obj/dsa_offloads.o obj/lzdatagen.o obj/pcg_basic.o  obj/pointer_chase.o src/protobuf/generated/src/protobuf/router.pb.h obj/router.pb.o obj/payload_gen.o obj/thread_utils.o obj/numa_mem.o obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a obj/stats.o
	$(CXX) -std=c++17 -march=native -O3 -o $@ $^ -DEXETIME $(INCLUDES) -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config -lz -lcrypto -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a -fpermissive
dmd: src/dmd.cpp obj/racer_opts.o obj/iaa_offloads.o obj/dsa_offloads.o obj/lzdatagen.o obj/pcg_basic.o  obj/pointer_chase.o src/protobuf/generated/src/protobuf/router.pb.h obj/router.pb.o obj/payload_gen.o obj/thread_utils.o obj/numa_mem.o obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a obj/stats.o
	$(CXX) -std=c++17 -march=native -O3 -o $@ $^ -DEXETIME $(INCLUDES) -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config -lz -lcrypto -lboost_coroutine -lboost_context -lpthread -lprotobuf -lpthread $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a -fpermissive

clean:
	rm -f obj/*.o obj/kernels/*.a obj/router.pb.o
	rm -rf ufh mmp ddh mg dmd dg
	rm -rf src/protobuf/generated/*