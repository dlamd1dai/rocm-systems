# ########################################################################
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Unit + functional tests for the GIN-SDMA collective designs
# (Broadcast / AllGather / AllToAll, deviceImpl == 3).
#
#   * gin_sdma_policy_test  - host-only GoogleTest binary covering the pure
#                             tier-selection / threshold / chunk / requirement
#                             logic in ../src/gin_sdma_collective_policy.h to
#                             high line+branch coverage. No GPU required. Runs
#                             under ctest label "unit".
#   * gin_sdma_gpu_<coll>   - ctest cases (label "gpu_functional") that drive the
#                             real device kernels across every tier via the
#                             *_perf binaries with the built-in correctness
#                             check. Need GPUs + MPI; opt-in.
# ########################################################################

# ---------------------------------------------------------------------------
# Host GoogleTest: pure policy logic (no GPU)
# ---------------------------------------------------------------------------
find_package(GTest QUIET)
if (NOT GTest_FOUND AND NOT TARGET gtest_main)
  message(STATUS "gin-sdma tests: system GTest not found, fetching googletest")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(gin_sdma_policy_test gin_sdma_policy_test.cpp)
# Pure host C++; keep it off the HIP device-compilation path even when the
# project compiler is hipcc/amdclang++ (GIN_SDMA_HOST_ONLY drops the
# __host__ __device__ attributes so no offload arch / device codegen is needed).
set_source_files_properties(gin_sdma_policy_test.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(gin_sdma_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(gin_sdma_policy_test PRIVATE GIN_SDMA_HOST_ONLY)

if (TARGET GTest::gtest_main)
  target_link_libraries(gin_sdma_policy_test PRIVATE GTest::gtest_main)
else()
  target_link_libraries(gin_sdma_policy_test PRIVATE gtest_main)
endif()

set_target_properties(gin_sdma_policy_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

option(ENABLE_COVERAGE "Instrument gin-sdma host unit tests for gcov/llvm-cov" OFF)
if (ENABLE_COVERAGE)
  target_compile_options(gin_sdma_policy_test PRIVATE -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(gin_sdma_policy_test PRIVATE -fprofile-arcs -ftest-coverage)

  # `make gin_sdma_coverage` runs the tests and prints a line+branch report for
  # the shared policy header, gating on GIN_SDMA_COVERAGE_MIN_PCT (default 95).
  # amdclang/hipcc emit gcov data readable by `llvm-cov gcov`; g++/gcc use gcov.
  set(_gcov_tool "gcov")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_program(LLVM_COV_EXE NAMES llvm-cov
                 HINTS ${ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin ENV ROCM_PATH)
    if (LLVM_COV_EXE)
      set(_gcov_tool "${LLVM_COV_EXE} gcov")
    endif()
  endif()
  set(GIN_SDMA_COVERAGE_MIN_PCT "95" CACHE STRING
      "Minimum line+branch % for the gin_sdma_coverage gate")
  add_custom_target(gin_sdma_coverage
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/gin_sdma_coverage.sh
            $<TARGET_FILE:gin_sdma_policy_test>
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/gin_sdma_collective_policy.h
            "${_gcov_tool}"
            ${GIN_SDMA_COVERAGE_MIN_PCT}
    DEPENDS gin_sdma_policy_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "GIN-SDMA policy coverage (gcov / llvm-cov gcov)")
endif()

add_test(NAME gin_sdma_policy_test COMMAND gin_sdma_policy_test)
set_tests_properties(gin_sdma_policy_test PROPERTIES LABELS "unit")

# ---------------------------------------------------------------------------
# GPU functional matrix (opt-in; needs GPUs + MPI + the *_perf binaries)
# ---------------------------------------------------------------------------
option(BUILD_GPU_FUNCTIONAL_TESTS
       "Register GIN-SDMA GPU functional ctest cases (need GPUs + MPI)" OFF)
if (BUILD_GPU_FUNCTIONAL_TESTS)
  set(GIN_SDMA_TEST_NP "8" CACHE STRING "MPI ranks for GIN-SDMA GPU functional tests")
  set(GIN_SDMA_MPI_LAUNCHER "mpirun" CACHE STRING "MPI launcher for GPU functional tests")
  foreach(coll broadcast all_gather alltoall)
    add_test(NAME gin_sdma_gpu_${coll}
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/gin_sdma_gpu_functional.sh
              ${coll} ${CMAKE_BINARY_DIR} ${GIN_SDMA_TEST_NP} ${GIN_SDMA_MPI_LAUNCHER})
    set_tests_properties(gin_sdma_gpu_${coll} PROPERTIES
      LABELS "gpu_functional" TIMEOUT 1200)
  endforeach()
  # The functional cases run the perf binaries; make sure they are built first.
  # (ctest does not build; callers should `make` before `ctest`.)
endif()
