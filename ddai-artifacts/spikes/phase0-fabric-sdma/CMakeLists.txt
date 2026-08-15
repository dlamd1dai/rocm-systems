cmake_minimum_required(VERSION 3.21)
project(phase0_fabric_sdma LANGUAGES CXX)

# Build the .cpp files (which contain HIP kernels) with hipcc:
#   cmake -S . -B build -DCMAKE_CXX_COMPILER=hipcc
# Optionally target a specific arch:
#   -DCMAKE_CXX_FLAGS="--offload-arch=gfx1250"
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip QUIET)

# --- Detect the HIP fabric handle API (mirrors projects/rccl CMake check) ---
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <hip/hip_runtime_api.h>
int main() {
  hipMemFabricHandle_t h;
  hipMemAllocationHandleType t = hipMemHandleTypeFabric;
  hipMemGenericAllocationHandle_t g = 0;
  int s = 0;
  (void)hipDeviceGetAttribute(&s, hipDeviceAttributeHandleTypeFabricSupported, 0);
  (void)hipMemImportFromShareableHandle(&g, (void*)&h, t);
  (void)hipMemExportToShareableHandle((void*)&h, g, t, 0);
  return 0;
}" SPIKE_HAS_FABRIC)

add_library(spike_common INTERFACE)
target_include_directories(spike_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
if(hip_FOUND)
  target_link_libraries(spike_common INTERFACE hip::device)
endif()
if(SPIKE_HAS_FABRIC)
  target_compile_definitions(spike_common INTERFACE SPIKE_HAS_FABRIC)
  message(STATUS "Fabric handle API: AVAILABLE (SPIKE_HAS_FABRIC)")
else()
  message(WARNING "Fabric handle API: NOT available in this ROCm build; fabric tests will report skip.")
endif()

add_executable(probe_fabric_caps probe_fabric_caps.cpp)
target_link_libraries(probe_fabric_caps PRIVATE spike_common)

add_executable(sdma_over_fabric_singlenode sdma_over_fabric_singlenode.cpp)
target_link_libraries(sdma_over_fabric_singlenode PRIVATE spike_common)

option(SPIKE_ENABLE_MPI "Build the multi-node MPI fabric test" OFF)
if(SPIKE_ENABLE_MPI)
  find_package(MPI REQUIRED)
  add_executable(multinode_fabric_mpi multinode_fabric_mpi.cpp)
  target_link_libraries(multinode_fabric_mpi PRIVATE spike_common MPI::MPI_CXX)
  target_compile_definitions(multinode_fabric_mpi PRIVATE USE_MPI)
endif()
