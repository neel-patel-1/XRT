cmake_minimum_required(VERSION 3.28)
project(XMPPass)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)

include(AddLLVM)

# Use LLVM's preprocessor definitions, include directories, and library search
# paths.
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_llvm_pass_plugin(XMPPass
    # List your source files here.
    XMPPass.cpp
)