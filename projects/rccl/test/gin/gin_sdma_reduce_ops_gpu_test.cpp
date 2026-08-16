/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// On-GPU correctness tests for the GIN-SDMA AllReduce reduction ops in
// projects/rccl-tests/src/gin_sdma_reduce.h (namespace gin_sdma_reduce): the
// preOp / combine / postOp reduction the ReduceScatter half of the AllReduce
// design (deviceImpl 5/6) applies on device.
//
// Unlike the host companion (gin_sdma_reduce_ops_test.cpp, which models __half
// as a float carrier and compiles bf16/fp8 out), this test compiles the header
// UNCHANGED for the GPU (real __device__ intrinsics) with the REAL element
// types -- __half (hip_fp16), hip_bfloat16 (HAVE_BF16=1) and the fp8 e4m3 /
// e5m2 types rccl_float8 / rccl_bfloat8 (HAVE_FP8=1) -- and folds actual
// contributions in a kernel on hardware. That validates the one thing the host
// shim cannot: the exact low-precision *rounding* (narrow-on-every-step) that
// must bit-match rccl-tests' verifiable.cu oracle, using the real hardware
// f32<->f8/f16/bf16 conversions.
//
// Two independent validation layers per (op, type):
//   (A) hand-computed exact-value checks -- inputs whose reduced result is
//       exactly representable in the target type, asserted against a constant
//       derived by hand (independent of the code under test);
//   (B) a broad randomized cross-check -- the device fold (real hardware
//       conversions) must produce the SAME storage bits as an independent host
//       mirror of the documented narrow-per-step spec (host software / compiler
//       conversions). Agreement across the value space confirms the device
//       codegen and the hardware rounding both match the spec.
//
// The __global__ AllReduce kernels in all_reduce.cu themselves (which need a
// live GIN comm + LSA windows) remain covered end-to-end by the multi-GPU
// all_reduce_perf datacheck sweeps; this file is the standalone single-GPU
// unit test for the reduction math those kernels call.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// ---- bring in the device reduce header, unmodified, with the real types -----
#define ENABLE_DEVICE_API 1
#ifndef NCCL_VERSION
#define NCCL_VERSION(a, b, c) ((a) * 10000 + (b) * 100 + (c))
#endif
#ifndef NCCL_VERSION_CODE
#define NCCL_VERSION_CODE NCCL_VERSION(2, 30, 4)
#endif
#define HAVE_BF16 1
#define HAVE_FP8 1
enum ncclRedOp_t { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 };

#include "rccl_float8.h"   // rccl_float8 (e4m3) / rccl_bfloat8 (e5m2), host+device
// rccl_float8.h declares `extern bool rccl_float8_useFnuz;` on the hip_fp8 path;
// provide the definition here so this standalone test links without librccl.
bool rccl_float8_useFnuz = false;

#include "gin_sdma_reduce.h"

using namespace gin_sdma_reduce;

namespace {

// --------------------------- type <-> float helpers --------------------------
// Overloaded host+device converters so the same fold code works for every T.
__host__ __device__ inline float toFloat(int v)          { return (float)v; }
__host__ __device__ inline float toFloat(int64_t v)      { return (float)v; }
__host__ __device__ inline float toFloat(float v)        { return v; }
__host__ __device__ inline float toFloat(double v)       { return (float)v; }
__host__ __device__ inline float toFloat(__half v)       { return __half2float(v); }
__host__ __device__ inline float toFloat(hip_bfloat16 v) { return static_cast<float>(v); }
__host__ __device__ inline float toFloat(rccl_float8 v)  { return static_cast<float>(v); }
__host__ __device__ inline float toFloat(rccl_bfloat8 v) { return static_cast<float>(v); }

template <class T> __host__ __device__ inline T fromFloat(float f);
template <> __host__ __device__ inline int          fromFloat<int>(float f)          { return (int)f; }
template <> __host__ __device__ inline int64_t      fromFloat<int64_t>(float f)      { return (int64_t)f; }
template <> __host__ __device__ inline float        fromFloat<float>(float f)        { return f; }
template <> __host__ __device__ inline double       fromFloat<double>(float f)       { return (double)f; }
template <> __host__ __device__ inline __half       fromFloat<__half>(float f)       { return __float2half(f); }
template <> __host__ __device__ inline hip_bfloat16 fromFloat<hip_bfloat16>(float f) { return hip_bfloat16(f); }
template <> __host__ __device__ inline rccl_float8  fromFloat<rccl_float8>(float f)  { return rccl_float8(f); }
template <> __host__ __device__ inline rccl_bfloat8 fromFloat<rccl_bfloat8>(float f) { return rccl_bfloat8(f); }

// Raw storage bits of a value (<= 8 bytes for every supported T), for the exact
// device-vs-host bit-match comparison in layer (B).
template <class T> __host__ __device__ inline uint64_t toBits(T v) {
  uint64_t b = 0;
  memcpy(&b, &v, sizeof(T));
  return b;
}

// -------------------------- the fold under test ------------------------------
// One reduction problem = rankN contributions folded in ascending source order,
// exactly as the ReduceScatter half of the AllReduce kernel drives the ops:
//   acc = preOp(in[0]); acc = combine(acc, preOp(in[r]))...; out = postOp(acc).
template <class T> __device__ inline T foldOne(int op, const float* f, int rankN) {
  T acc = preOp<T>(op, fromFloat<T>(f[0]), rankN);
  for (int r = 1; r < rankN; ++r) {
    T v = preOp<T>(op, fromFloat<T>(f[r]), rankN);
    acc = combine<T>(op, acc, v);
  }
  return postOp<T>(op, acc, rankN);
}

template <class T>
__global__ void foldKernel(const float* in, int problems, int rankN, int op, uint64_t* outBits) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= problems) return;
  outBits[p] = toBits<T>(foldOne<T>(op, in + (size_t)p * rankN, rankN));
}

// ------------------- host mirror of the documented spec (layer B) ------------
// Independent host reimplementation of preOp/combine/postOp following the
// header's contract (narrow-on-every-step for low-precision; native arithmetic
// for int/float/double; avg premultiplies on the floating branch and integer-
// divides on the integral branch). Uses the HOST conversion path (software /
// compiler), a different implementation from the device's hardware v_cvt, so
// bitwise agreement is a genuine cross-check rather than a tautology.
template <class T> inline bool isLowPrec() {
  return std::is_same<T, __half>::value || std::is_same<T, hip_bfloat16>::value ||
         std::is_same<T, rccl_float8>::value || std::is_same<T, rccl_bfloat8>::value;
}

template <class T> inline T hcombine(int op, T a, T b) {
  if constexpr (std::is_integral<T>::value || std::is_same<T, float>::value ||
                std::is_same<T, double>::value) {
    // Native arithmetic in T (mirrors the generic rsum/rprod/rmin/rmax template).
    switch (op) {
      case ncclProd: return (T)(a * b);
      case ncclMax:  return a > b ? a : b;
      case ncclMin:  return a < b ? a : b;
      default:       return (T)(a + b);
    }
  } else {
    // low-precision: through float with narrow-per-step (mirrors the overloads).
    float fa = toFloat(a), fb = toFloat(b);
    switch (op) {
      case ncclProd: return fromFloat<T>(fa * fb);
      case ncclMax:  return fa > fb ? a : b;   // header keeps the operand (no narrow)
      case ncclMin:  return fa < fb ? a : b;
      default:       return fromFloat<T>(fa + fb);
    }
  }
}

template <class T> inline T hpreOp(int op, T x, int rankN) {
  if (op == ncclAvg && !std::is_integral<T>::value) {
    float factor = 1.0f / (float)rankN;      // matches avgFactor<T> (float intermediate)
    if (isLowPrec<T>()) {
      T fT = fromFloat<T>(factor);            // narrow factor to T (rprod overload narrows)
      return fromFloat<T>(toFloat(fT) * toFloat(x));
    }
    return fromFloat<T>(factor * toFloat(x)); // float branch
  }
  return x;
}

template <class T> inline T hpostOp(int op, T x, int rankN) {
  if (op == ncclAvg && std::is_integral<T>::value) {
    return fromFloat<T>((float)((int64_t)toFloat(x) / rankN));  // integer divide
  }
  return x;
}

template <class T> inline T hFoldOne(int op, const float* f, int rankN) {
  T acc = hpreOp<T>(op, fromFloat<T>(f[0]), rankN);
  for (int r = 1; r < rankN; ++r) {
    T v = hpreOp<T>(op, fromFloat<T>(f[r]), rankN);
    acc = hcombine<T>(op, acc, v);
  }
  return hpostOp<T>(op, acc, rankN);
}

// double needs the double-precision arithmetic path (avgFactor<double> and
// native +/*), so it gets its own mirror rather than routing through float.
inline double hFoldOneDouble(int op, const float* f, int rankN) {
  auto comb = [&](double a, double b) -> double {
    switch (op) {
      case ncclProd: return a * b;
      case ncclMax:  return a > b ? a : b;
      case ncclMin:  return a < b ? a : b;
      default:       return a + b;
    }
  };
  double factor = 1.0 / (double)rankN;
  auto pre = [&](double x) -> double { return (op == ncclAvg) ? factor * x : x; };
  double acc = pre((double)f[0]);
  for (int r = 1; r < rankN; ++r) acc = comb(acc, pre((double)f[r]));
  return acc;  // floating avg divides in preOp; postOp identity
}

// ----------------------------- test harness ---------------------------------
class ReduceOpsGpu : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    haveGpu_ = (e == hipSuccess && n > 0);
    if (haveGpu_) (void)hipSetDevice(0);
  }
  static bool haveGpu_;
};
bool ReduceOpsGpu::haveGpu_ = false;

#define REQUIRE_GPU()                                                    \
  do {                                                                   \
    if (!haveGpu_) GTEST_SKIP() << "no ROCm device visible";             \
  } while (0)

// Launch the device fold for a batch of problems and return the raw result
// bits, one per problem.
template <class T>
std::vector<uint64_t> runDeviceFold(int op, const std::vector<float>& inputs,
                                    int problems, int rankN) {
  float* dIn = nullptr;
  uint64_t* dOut = nullptr;
  EXPECT_EQ(hipMalloc(&dIn, inputs.size() * sizeof(float)), hipSuccess);
  EXPECT_EQ(hipMalloc(&dOut, (size_t)problems * sizeof(uint64_t)), hipSuccess);
  EXPECT_EQ(hipMemcpy(dIn, inputs.data(), inputs.size() * sizeof(float),
                      hipMemcpyHostToDevice), hipSuccess);
  int threads = 128, blocks = (problems + threads - 1) / threads;
  hipLaunchKernelGGL(foldKernel<T>, dim3(blocks), dim3(threads), 0, 0,
                     dIn, problems, rankN, op, dOut);
  EXPECT_EQ(hipGetLastError(), hipSuccess);
  EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
  std::vector<uint64_t> out((size_t)problems);
  EXPECT_EQ(hipMemcpy(out.data(), dOut, (size_t)problems * sizeof(uint64_t),
                      hipMemcpyDeviceToHost), hipSuccess);
  (void)hipFree(dIn);
  (void)hipFree(dOut);
  return out;
}

// Single-problem device fold, returning the result as float (for the exact
// hand-computed checks in layer A).
template <class T>
float deviceFoldScalar(int op, const std::vector<float>& f) {
  auto bits = runDeviceFold<T>(op, f, 1, (int)f.size());
  T v;
  memcpy(&v, &bits[0], sizeof(T));
  return toFloat(v);
}

// ---- layer (A): hand-computed exact-value checks (independent oracle) --------
// __half / bf16 sum & prod of small exactly-representable integers.
TEST_F(ReduceOpsGpu, ExactHalfBf16) {
  REQUIRE_GPU();
  std::vector<float> v = {1, 2, 3, 4};             // sum=10, prod=24, min=1, max=4
  EXPECT_FLOAT_EQ(deviceFoldScalar<__half>(ncclSum, v), 10.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<__half>(ncclProd, v), 24.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<__half>(ncclMax, v), 4.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<__half>(ncclMin, v), 1.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<__half>(ncclAvg, v), 2.5f);   // floating avg
  EXPECT_FLOAT_EQ(deviceFoldScalar<hip_bfloat16>(ncclSum, v), 10.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<hip_bfloat16>(ncclProd, v), 24.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<hip_bfloat16>(ncclAvg, v), 2.5f);
}

// fp8 e4m3 (rccl_float8) & e5m2 (rccl_bfloat8): use values whose result is
// exactly representable in the respective format.
TEST_F(ReduceOpsGpu, ExactFp8) {
  REQUIRE_GPU();
  std::vector<float> v = {1, 2, 1, 2};             // sum=6, min=1, max=2, avg=1.5
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_float8>(ncclSum, v), 6.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_float8>(ncclMax, v), 2.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_float8>(ncclMin, v), 1.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_float8>(ncclAvg, v), 1.5f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_bfloat8>(ncclSum, v), 6.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_bfloat8>(ncclMax, v), 2.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<rccl_bfloat8>(ncclAvg, v), 1.5f);
}

// float / double / integer exact reductions.
TEST_F(ReduceOpsGpu, ExactWideTypes) {
  REQUIRE_GPU();
  std::vector<float> v = {1, 2, 3, 4};
  EXPECT_FLOAT_EQ(deviceFoldScalar<float>(ncclSum, v), 10.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<float>(ncclProd, v), 24.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<float>(ncclAvg, v), 2.5f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<double>(ncclSum, v), 10.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<double>(ncclAvg, v), 2.5f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<int>(ncclSum, v), 10.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<int>(ncclProd, v), 24.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<int>(ncclAvg, v), 2.0f);   // integer avg: 10/4 = 2
  EXPECT_FLOAT_EQ(deviceFoldScalar<int64_t>(ncclSum, v), 10.0f);
  EXPECT_FLOAT_EQ(deviceFoldScalar<int64_t>(ncclAvg, v), 2.0f);
}

// ---- layer (B): broad randomized device-vs-host-spec bit-match --------------
// For a type T and op, generate many random problems, fold on device, and
// require each result's storage bits to equal the host mirror's. Validates the
// hardware narrow-per-step rounding matches the documented spec across the
// value space (the exact property that must bit-match verifiable.cu).
template <class T>
void randomBitMatch(int op, std::mt19937& rng, int problems, int rankN,
                    float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> inputs((size_t)problems * rankN);
  for (auto& x : inputs) x = dist(rng);
  auto dev = runDeviceFold<T>(op, inputs, problems, rankN);
  for (int p = 0; p < problems; ++p) {
    T ref = hFoldOne<T>(op, inputs.data() + (size_t)p * rankN, rankN);
    EXPECT_EQ(dev[p], toBits<T>(ref))
        << "op=" << op << " rankN=" << rankN << " problem=" << p
        << " device=" << toFloat(*reinterpret_cast<T*>(&dev[p]))
        << " host=" << toFloat(ref);
  }
}

template <class T>
void sweepAllOps(std::mt19937& rng, float lo, float hi) {
  const int ops[] = {ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg};
  for (int rankN : {2, 3, 4, 8}) {
    for (int op : ops) {
      // fp8 prod is intentionally not dispatched by the design; the rprod fp8
      // overload still exists, but its dynamic range makes randomized products
      // overflow/saturate frequently, so skip prod for fp8 here (sum/min/max/
      // avg give the meaningful rounding coverage). All other types test prod.
      if (op == ncclProd &&
          (std::is_same<T, rccl_float8>::value || std::is_same<T, rccl_bfloat8>::value))
        continue;
      randomBitMatch<T>(op, rng, /*problems=*/4096, rankN, lo, hi);
    }
  }
}

TEST_F(ReduceOpsGpu, RandomBitMatchHalf) {
  REQUIRE_GPU();
  std::mt19937 rng(0xA11CE);
  sweepAllOps<__half>(rng, -50.f, 50.f);
}

TEST_F(ReduceOpsGpu, RandomBitMatchBf16) {
  REQUIRE_GPU();
  std::mt19937 rng(0xB16);
  sweepAllOps<hip_bfloat16>(rng, -50.f, 50.f);
}

TEST_F(ReduceOpsGpu, RandomBitMatchFp8E4M3) {
  REQUIRE_GPU();
  std::mt19937 rng(0xF8E4);
  sweepAllOps<rccl_float8>(rng, -8.f, 8.f);
}

TEST_F(ReduceOpsGpu, RandomBitMatchFp8E5M2) {
  REQUIRE_GPU();
  std::mt19937 rng(0xF8E5);
  sweepAllOps<rccl_bfloat8>(rng, -16.f, 16.f);
}

TEST_F(ReduceOpsGpu, RandomBitMatchFloat) {
  REQUIRE_GPU();
  std::mt19937 rng(0xF10A7);
  sweepAllOps<float>(rng, -1000.f, 1000.f);
}

TEST_F(ReduceOpsGpu, RandomBitMatchIntegers) {
  REQUIRE_GPU();
  std::mt19937 rng(0x142);
  // Range kept tight so an 8-way product stays within int range (3^8 = 6561),
  // avoiding signed-overflow UB while still exercising native integer sum/prod/
  // min/max and the integer-divide avg path.
  sweepAllOps<int>(rng, -3.f, 3.f);
  sweepAllOps<int64_t>(rng, -3.f, 3.f);
}

// double uses the double-precision arithmetic path (avgFactor<double>), so it
// is cross-checked against the dedicated double host mirror.
TEST_F(ReduceOpsGpu, RandomBitMatchDouble) {
  REQUIRE_GPU();
  std::mt19937 rng(0xD0B1E);
  std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
  const int ops[] = {ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg};
  for (int rankN : {2, 3, 4, 8}) {
    for (int op : ops) {
      const int problems = 4096;
      std::vector<float> inputs((size_t)problems * rankN);
      for (auto& x : inputs) x = dist(rng);
      auto dev = runDeviceFold<double>(op, inputs, problems, rankN);
      for (int p = 0; p < problems; ++p) {
        double ref = hFoldOneDouble(op, inputs.data() + (size_t)p * rankN, rankN);
        EXPECT_EQ(dev[p], toBits<double>(ref)) << "double op=" << op << " rankN=" << rankN;
      }
    }
  }
}

}  // namespace
