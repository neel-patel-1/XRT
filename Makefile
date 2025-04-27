ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CXX=g++
INCLUDES=-I$(ROOT_DIR)/third-party/idxd-config/test/algorithms -I$(ROOT_DIR)/third-party/idxd-config/test -I$(ROOT_DIR)/include -I$(ROOT_DIR)/third-party/idxd-config/accfg -I$(ROOT_DIR)/third-party/idxd-config -I$(ROOT_DIR) -I$(ROOT_DIR)/third-party/ipp-crypto/_build/RELEASE/include -I$(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/include -I$(ROOT_DIR)/include/inline
LLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm/

all: mmp ufh ddh mg dg dmd

kernel_objs= obj/kernels/ch3_hash.a obj/kernels/dotproduct.a obj/kernels/gather.a obj/kernels/matmul_histogram.a obj/kernels/sequential_writer.a obj/pointer_chase.o obj/payload_gen.o obj/lzdatagen.o obj/pcg_basic.o obj/router.pb.o
kernel_libs = -lz -lprotobuf $(ROOT_DIR)/third-party/ipp-crypto/_build/.build/RELEASE/lib/libippcp.a

idxd_objs= obj/iaa_offloads.o obj/dsa_offloads.o
idxd_libs = -Wl,-rpath,$(ROOT_DIR)/third-party/idxd-config/install/lib64 -L$(ROOT_DIR)/third-party/idxd-config/test -liaa -L$(ROOT_DIR)/third-party/idxd-config/install/lib64 -laccel-config

fcontext_objs= obj/jump_x86_64_sysv_elf_gas.o obj/make_x86_64_sysv_elf_gas.o obj/ontop_x86_64_sysv_elf_gas.o obj/context_fast.o

obj/jump_x86_64_sysv_elf_gas.o: src/jump_x86_64_sysv_elf_gas.S
	$(CXX)  -c -o $@ $^  -I./include

obj/make_x86_64_sysv_elf_gas.o: src/make_x86_64_sysv_elf_gas.S
	$(CXX)  -c -o $@ $^  -I./include

obj/ontop_x86_64_sysv_elf_gas.o: src/ontop_x86_64_sysv_elf_gas.S
	$(CXX)  -c -o $@ $^  -I./include

obj/context_fast.o: src/context_fast.S
	$(CXX)  -c -o $@ $^  -I./include

obj/kernels/%.a: src/kernels/%.cpp
	$(CXX) -std=c++17 -mavx512f -march=native -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/include

obj/%.o: src/%.cpp
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ $(INCLUDES) -fpermissive

obj/router.pb.o: src/protobuf/generated/src/protobuf/router.pb.cc
	$(CXX) -std=c++17 -O3 -fPIC -c $< -o $@ -I$(ROOT_DIR)/src/protobuf/generated/

src/protobuf/generated/src/protobuf/router.pb.h: src/protobuf/router.proto
	protoc --cpp_out=./src/protobuf/generated $<

src/protobuf/generated/src/protobuf/router.pb.cc: src/protobuf/router.proto
	protoc --cpp_out=./src/protobuf/generated $<


# Fig 3
src/XMPPass/build/XMPPass.so: src/XMPPass/XMPPass.cpp
	mkdir -p src/XMPPass/build
	cd src/XMPPass/build && cmake -DLLVM_DIR=$(LLVM_DIR) .. && make
exe_time: src/baseline.cpp $(idxd_objs) $(kernel_objs) $(fcontext_objs) obj/numa_mem.o obj/stats.o obj/thread_utils.o
	$(CXX) -std=c++17 -march=native -O3 -o $@ $^ -DEXETIME $(INCLUDES)  $(idxd_libs) $(kernel_libs)

# Fig 9
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
	rm -rf src/XMPPass/build/*