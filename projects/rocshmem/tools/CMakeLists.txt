###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################
# rocshmem_info — host-only diagnostic binary
###############################################################################

# Do not link roc::rocshmem: its PUBLIC -fgpu-rdc interface triggers HIP device
# link / LTO that records RHEL-style /usr/lib64/libc.so prerequisites and breaks
# on Debian/Ubuntu (Dockerfile-rccl-gin-gda-sdma, Ubuntu 24.04).

set(_rocshmem_info_sources
  ${CMAKE_SOURCE_DIR}/src/build_info.cpp
  ${CMAKE_SOURCE_DIR}/src/envvar.cpp
  ${CMAKE_SOURCE_DIR}/tools/rocshmem_info_alloc_stub.cpp
)

if(USE_GDA)
  list(APPEND _rocshmem_info_sources
    ${CMAKE_SOURCE_DIR}/src/gda/topology.cpp
    ${CMAKE_SOURCE_DIR}/src/gda/ibv_wrapper.cpp
    ${CMAKE_SOURCE_DIR}/src/memory/memory_allocator.cpp
  )
endif()

add_library(rocshmem_info_host STATIC ${_rocshmem_info_sources})

target_include_directories(rocshmem_info_host
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/rocshmem
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(rocshmem_info_host PRIVATE -fno-gpu-rdc)

target_link_libraries(rocshmem_info_host
  PUBLIC
    hip::host
    hsa-runtime64::hsa-runtime64
    Threads::Threads
    dl
)

if(USE_GDA)
  target_link_libraries(rocshmem_info_host PUBLIC numa::numa)
endif()

add_executable(rocshmem_info rocshmem_info.cpp)

target_include_directories(rocshmem_info
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(rocshmem_info PRIVATE -fno-gpu-rdc)

target_link_libraries(rocshmem_info PRIVATE rocshmem_info_host)

set_target_properties(rocshmem_info PROPERTIES LINKER_LANGUAGE CXX)
