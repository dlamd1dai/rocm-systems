/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#define NCCL_TESTS_VERSION "2.18.3"

// Pure (no-GPU) policy logic shared by the GIN-SDMA collective designs and their
// host unit tests. Included first so the threshold/tier helpers below can
// delegate to it.
#include "gin_sdma_collective_policy.h"
#include "rccl/rccl.h"
// nccl_device.h provides the device-API public types referenced below
// (ncclCommProperties_t, full ncclDevCommRequirements, NCCL_GIN_TYPE_NONE, ...).
// As of NCCL 2.29 these types are referenced unconditionally by the test engine
// signature and per-collective DevCommRequirements helpers, so the include must
// not require -DENABLE_DEVICE_API on 2.29+.
#if (defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)) \
    || NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
#include "nccl_device.h"
#endif
#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>
#include "nccl1_compat.h"
#include "rccl_compat.h"  // Weak symbols forward declarations
#include "timer.h"
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <map>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test CUDA failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    return testCudaError;                           \
  }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d "           \
           "'%s / %s'\n",                           \
           hostname,__FILE__,__LINE__,              \
           ncclGetErrorString(res),                 \
           ncclGetLastError(NULL));                 \
    return testNcclError;                           \
  }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    return testNcclError;                           \
  }                                                 \
} while(0)
#endif

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testTimeout = 4,
  testNotImplemented = 5,
  testInvalidUsage = 6,
  testNumResults = 7, // Must be last
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s pid %d: Test failure %s:%d\n",   \
         hostname, getpid(),                        \
        __FILE__,__LINE__);                         \
    return r;                                       \
  }                                                 \
} while(0)

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, size_t eltSize, int nranks);
  testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type,
      ncclRedOp_t op, int root, int rep, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset,
      size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implIndex, void* bias);
  testResult_t (*getAlgoProtoChannels)(ncclComm_t comm, size_t count, ncclDataType_t type, int* algo, int* proto, int* nchannels);
  testResult_t (*getSymkInfo)(ncclComm_t comm, size_t count, ncclDataType_t type, ncclRedOp_t op, int* algo, int* proto, int* nchannels);
  // Optional device-side (in-kernel wall_clock64) timing hook. Non-null only for
  // collectives that implement it (currently AllToAll and AllReduce). Driven by BenchTime via
  // NCCL_GIN_ANVIL_DEVICE_TIMING (or legacy NCCL_GIN_ANVIL_A2A_DEVICE_TIMING)
  // (0=off, 1=augment, 2=device-time-only):
  //   - outDeltaSec == nullptr (mode 1): prints an extra device-only
  //     latency/busbw line alongside the normal graph/hipEvent numbers (report,
  //     not replace).
  //   - outDeltaSec != nullptr (mode 2): stores the measured per-iteration
  //     device latency (seconds, max across ranks) in *outDeltaSec and prints a
  //     short context annotation; BenchTime then reports that as THE metric,
  //     having skipped the graph/hipEvent timed loop.
  // Other collectives leave it nullptr via aggregate initialization, so no other
  // struct needs to change.
  testResult_t (*deviceTime)(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double* outDeltaSec);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;
extern struct testColl alltoAllTest;

class Reporter {
  public:
    Reporter(std::string fileName, std::string outputFormat);
    ~Reporter() { if (_outputValid) { _out.close(); } };
    void setParameters(const size_t numCycle, const char* name, const char* typeName, const char* opName);
    void addResult(int gpusPerRank, int ranksPerNode, int totalRanks, size_t numBytes, int inPlace, double timeUsec, double algBw, double busBw, int64_t wrongElts = -1);
    void writeFile();

  private:
    bool isMainThread();
    template<typename T> std::pair<std::string, std::string> makeValueKeyPair(T v, std::string k) { return std::make_pair(std::to_string(v), k); };
    template <> std::pair<std::string, std::string> makeValueKeyPair<std::string>(std::string v, std::string k) { return std::make_pair("\"" + v + "\"", k); };

    bool _outputValid = false;
    std::ofstream _out;
    std::string _outputFormat;
    size_t _numCycle = 0;
    std::string _collectiveName;
    std::string _typeName;
    std::string _opName;
    std::vector<std::vector<std::pair<std::string, std::string>>> _outputData;
};

struct testEngine {
  void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
  testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
      const char* typeName, ncclRedOp_t op, const char* opName);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  /* Optional; called from initComms after common fields are set on ncclConfig_t. */
  void (*initCommConfig)(ncclConfig_t* config);
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  testResult_t (*getDevCommRequirements)(int deviceImpl, ncclDevCommRequirements* reqs, ncclCommProperties_t* commProperties);
#elif defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  bool (*getDevCommRequirements)(int deviceImpl, ncclDevCommRequirements* reqs);
#endif
};

extern struct testEngine ncclTestEngine;

struct threadArgs {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  int enable_out_of_place;
  int enable_in_place;
  int enable_cache_flush;
  int enable_rotating_tensor;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  ncclUniqueId ncclId;
  ncclComm_t* comms;
#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  ncclDevComm* devComms;
#endif
  cudaStream_t* streams;
  void** bias;

  void** expected;
  size_t expectedBytes;
  int* errors;
  double* bw;
  int* bw_count;

  int reportErrors;

  struct testColl* collTest;

  Reporter* reporter;

  int64_t* initGpuMem;
  int64_t* bufferMemory;
  int64_t* devMemUsed;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  void** sendRegHandles;
  void** recvRegHandles;
  void** biasRegHandles;
#endif
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

// Provided by common.cu
extern void Barrier(struct threadArgs* args);
extern testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op,  const char* opName, int root);
extern testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks);
extern testResult_t InitDataApplyBias(void* expected, void* bias, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op);
extern testResult_t InitData(void* data, const size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks, const int rank);
extern testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, void **bias);

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '\0') {
      return;
    }
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHash(const char* string, size_t n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash, strlen(hostHash));
}

#if NCCL_MAJOR >= 2 && RCCL_BFLOAT16 == 1
#define HAVE_BF16 1
#else
#define HAVE_BF16 0
#endif
#if NCCL_MAJOR >= 2 && RCCL_FLOAT8 == 1
#define HAVE_FP8 1
#else
#define HAVE_FP8 0
#endif

#if NCCL_MAJOR >= 2
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    #undef HAVE_BF16
    #define HAVE_BF16 1
    #if defined(__CUDA_FP8_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,24,0)
      #undef HAVE_FP8
      #define HAVE_FP8 1
    #endif
  #endif
#endif

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#if HAVE_FP8
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
#endif
#endif
      return 1;
    case ncclHalf:
#if HAVE_BF16
    case ncclBfloat16:
#endif
    //case ncclFloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    //case ncclInt32:
    case ncclUint32:
    //case ncclFloat32:
#endif
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
    //case ncclFloat64:
      return 8;
    default: return 0;
  }
}

extern int test_ncclVersion; // init'd with ncclGetVersion()
typedef enum { ncclCoarse        = 0,
               ncclFine          = 1,
               ncclHost          = 2,
               ncclManaged       = 3,
               nccl_NUM_MTYPES   = 4 } ncclMemoryType_t;
extern const char *test_memorytypes[nccl_NUM_MTYPES];
extern int deviceCtaCount; // number of CTAs for device implementation
extern int deviceImpl;     // selected -D device implementation (0 = host); lets per-coll
                           // RunTest skip op/type combos unimplemented on the device path
extern size_t maxBytes;    // largest message the run will exercise (from -e); used
                           // by device-kernel scratch-window sizing (ReduceScatter)
constexpr int test_opNumMax = (int)ncclNumOps + (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) ? 1 : 0);
extern int test_opnum;
extern int test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char *test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[];
extern const char *test_opnames[];

static int ncclstringtotype(char *str) {
    for (int t=0; t<test_typenum; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[ncclFloat]);
    return ncclFloat;
}

static int ncclstringtoop (char *str) {
    for (int o=0; o<test_opnum; o++) {
      if (strcmp(str, test_opnames[o]) == 0) {
        return o;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[ncclSum]);
    return ncclSum;
}

static int ncclstringtoroot (char *str) {
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    return strtol(str, NULL, 0);
}

static int ncclstringtomtype (char *str) {
    for (int o=0; o<nccl_NUM_MTYPES; o++) {
      if (strcmp(str, test_memorytypes[o]) == 0) {
        return o;
      }
    }
    printf("invalid memorytype %s, defaulting to %s .. \n", str, test_memorytypes[ncclCoarse]);
    return ncclCoarse;
}

extern int is_main_proc;
extern thread_local int is_main_thread;

// Sentinel meaning "no per-collective override; use the device/backend value
// (i.e. rsCtx->sdmaThreshold, populated from NCCL_GIN_ANVIL_SDMA_THRESHOLD)".
#define TEST_SDMA_THRESHOLD_UNSET (gin_sdma::kThresholdUnset)

// Parse a per-collective LSA<->GIN threshold env var (bytes; optional K/M/G
// suffix). Returns TEST_SDMA_THRESHOLD_UNSET when unset/empty/unparseable so
// the kernel falls back to the shared NCCL_GIN_ANVIL_SDMA_THRESHOLD value.
// Thin getenv() wrapper over the pure, unit-tested gin_sdma::parseSize().
static inline size_t testParseSdmaThresholdEnv(const char* name) {
  return gin_sdma::parseSize(getenv(name));
}

// Resolve a collective's LSA<->GIN threshold with the fallback chain:
//   1. the collective-specific env var (collVar), if set;
//   2. the shared NCCL_GIN_ANVIL_SDMA_THRESHOLD, if explicitly set (keeps the
//      global force knob, e.g. BC-D4's THRESHOLD=0, working);
//   3. the collective's data-driven default (collDefault).
// Always returns a concrete value, so callers pass it straight to the kernel.
static inline size_t testResolveSdmaThreshold(const char* collVar, size_t collDefault) {
  return gin_sdma::resolveThreshold(gin_sdma::parseSize(getenv(collVar)),
                                    gin_sdma::parseSize(getenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD")),
                                    collDefault);
}

#if defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// Overflow-safe gin.put for the Anvil-SDMA backend. The SDMA linear-copy count
// field is 30 bits and 1-based (count = bytes-1), so the largest single packet
// is exactly 2^30 = 1 GiB; a put of >1 GiB silently truncates and corrupts data
// (a 2 GiB transfer copies only 1 GiB). Split the transfer into
// <=gin_sdma::kGinPutMaxBytes (1 GiB, the HW max) segments and carry the caller's
// remote action (e.g. SignalInc) ONLY on the final segment: the SDMA queue is
// in-order, so a single signal still correctly means "the whole message has
// landed" and per-message signal accounting (waitSignal counts) is unchanged. A
// <=1 GiB message is a single put with no extra overhead. Threads still each own
// a disjoint (peer, offset) tuple, so the inner segmentation is race-free.
template <typename RemoteAction>
__device__ __forceinline__ void ginPutChunked(
    ncclGin& gin, ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOff,
    ncclWindow_t srcWin, size_t srcOff,
    size_t bytes, RemoteAction finalAction) {
  const size_t kMax = gin_sdma::kGinPutMaxBytes;
  size_t off = 0;
  do {
    const size_t rem = bytes - off;
    const size_t seg = rem > kMax ? kMax : rem;
    if (off + seg >= bytes) {
      // Final (or only) segment carries the signal / remote action.
      gin.put(team, peer, dstWin, dstOff + off, srcWin, srcOff + off, seg, finalAction);
    } else {
      gin.put(team, peer, dstWin, dstOff + off, srcWin, srcOff + off, seg);
    }
    off += seg;
  } while (off < bytes);
}

template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
  return testSuccess;
}

// Variant that passes a per-collective LSA<->GIN threshold override as the last
// kernel argument. Used by AllGather/Broadcast GIN kernels; AlltoAll keeps the
// base launcher (its LSA-vs-GIN split is topology-based, not size-based).
template <typename F>
testResult_t testLaunchDeviceKernelThreshold(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride);
  return testSuccess;
}

// Variant that also forwards an LL (low-latency, packed data+flag) handle as the
// last kernel argument. Used by the AllGather GIN kernel for its tiny-message
// LL fast path; the handle is a small POD ({bufHandle, nSlots}) assigned during
// ncclDevCommCreate.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdLL(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, llHandle);
  return testSuccess;
}

// Like ...ThresholdLL, but the caller picks the grid (CTA count) per call instead
// of using the global deviceCtaCount. Used by AllToAll's size-adaptive LSA tier
// (F1): the CTA count grows with the per-peer chunk to scale xGMI egress like the
// host RING's channel parallelism, while the SDMA tier launches a small grid. The
// requested grid must be <= the allocated lsaBarrier count (see a2aDevReqs), which
// is sized for kA2aLsaMaxCtas; launching fewer CTAs than allocated is always safe.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdLLCtas(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int gridCtas) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, llHandle);
  return testSuccess;
}

// Like ...ThresholdLLCtas, but also forwards a trailing int (caller-picked grid +
// an int flag). Used by AllToAll's LSA tier to select its cross-rank sync mode at
// runtime (0 = LSA barriers, 1 = none [diagnostic ceiling], 2 = point-to-point
// ready/done flags).
template <typename F>
testResult_t testLaunchDeviceKernelThresholdLLCtasFlag(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int gridCtas, int flag, ncclDevResourceHandle_t flagBuf) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  if (gridCtas < 1) gridCtas = 1;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, llHandle, flag, flagBuf);
  return testSuccess;
}

// Variant that forwards both an LL handle and a trailing int flag as the last
// two kernel arguments. Used by the Scatter GIN kernel to toggle its LSA-tier
// root fan-out layout (peer-interleaved vs sequential) at runtime from an env.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdLLFlag(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclLLA2AHandle llHandle, int flag) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, llHandle, flag);
  return testSuccess;
}

// Variant for the reduction collectives (ReduceScatter): forwards the LSA<->GIN
// threshold, the reduction op (as an int -- the kernel switches on it via
// gin_sdma_reduce), and a resource-buffer scratch-window handle for the large
// put-partials tier. The handle is a small POD (uint32_t) assigned during
// ncclDevCommCreate; 0 means no scratch was configured.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdScratch(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclDevResourceHandle scratchHandle) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, (int)op, scratchHandle);
  return testSuccess;
}

// Single-launch variant with an EXPLICIT grid size (gridCtas) instead of the global
// deviceCtaCount. Used by the GIN-SDMA AllReduce -D 5 to pick a size-adaptive CTA count
// (few CTAs for small messages, more for large; see arTunedGridCtas). gridCtas must be
// <= the barrier/signal slot count registered in AllReduceGetDevCommRequirements.
template <typename F>
testResult_t testLaunchDeviceKernelThresholdScratchCtas(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclDevResourceHandle scratchHandle, int gridCtas, size_t oneShotThresholdOverride) {
  if (kernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, (int)op, scratchHandle, oneShotThresholdOverride);
  return testSuccess;
}

// Two-launch variant (AllReduce -D 6) with an EXPLICIT grid size: launches a ReduceScatter
// kernel then an AllGather kernel back-to-back on the SAME stream, so the RS->AG boundary
// is the kernel-launch boundary. Both kernels share the reduction-collective signature.
template <typename F>
testResult_t testLaunchDeviceKernelAR2SplitCtas(F rsKernel, F agKernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, ncclDevResourceHandle scratchHandle, int gridCtas) {
  if (rsKernel == nullptr || agKernel == nullptr) return testNotImplemented;
  ncclDevComm* devComm = (ncclDevComm*)comm;
  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  rsKernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, (int)op, scratchHandle);
  agKernel<<<gridCtas, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm, sdmaThresholdOverride, (int)op, scratchHandle);
  return testSuccess;
}

#define SPECIALIZE_KERNEL(kernel, type, op) \
  ( op != ncclSum ? nullptr : \
   type == ncclInt8 ? kernel<int8_t> : \
   type == ncclUint8 ? kernel<uint8_t> : \
   type == ncclInt32 ? kernel<int32_t> : \
   type == ncclUint32 ? kernel<uint32_t> : \
   type == ncclInt64 ? kernel<int64_t> : \
   type == ncclUint64 ? kernel<uint64_t> : \
   type == ncclFloat16 ? kernel<half> : \
   type == ncclFloat32 ? kernel<float> : \
   type == ncclFloat64 ? kernel<double> : \
   nullptr \
  )

// Op-aware dispatch for the reduction collectives (ReduceScatter). Unlike
// SPECIALIZE_KERNEL (which forces op==ncclSum), this selects kernel<T> across the
// full element-type set for any of the supported built-in reduction ops
// (sum/prod/max/min/avg). PreMulSum ("mulsum", a created op handle >= ncclNumOps)
// is NOT supported (deferred) -> nullptr -> testNotImplemented; fp8 prod is
// excluded too (matches the ReduceScatterRunTest skip). The reduction op itself
// is passed to the kernel at launch (runtime switch), so the template varies
// only on T.
#if HAVE_BF16
#define RS_BF16_CASE(kernel, type) (type) == ncclBfloat16 ? kernel<hip_bfloat16> :
#else
#define RS_BF16_CASE(kernel, type)
#endif
#if HAVE_FP8
#define RS_FP8_CASE(kernel, type) (type) == ncclFloat8e4m3 ? kernel<rccl_float8> : (type) == ncclFloat8e5m2 ? kernel<rccl_bfloat8> :
#else
#define RS_FP8_CASE(kernel, type)
#endif
#define SPECIALIZE_REDUCE_KERNEL(kernel, type, op) \
  ( (int)(op) >= (int)ncclNumOps ? nullptr : \
    (((op) == ncclProd && ((type) == ncclFloat8e4m3 || (type) == ncclFloat8e5m2)) ? nullptr : \
     (type) == ncclInt8 ? kernel<int8_t> : \
     (type) == ncclUint8 ? kernel<uint8_t> : \
     (type) == ncclInt32 ? kernel<int32_t> : \
     (type) == ncclUint32 ? kernel<uint32_t> : \
     (type) == ncclInt64 ? kernel<int64_t> : \
     (type) == ncclUint64 ? kernel<uint64_t> : \
     (type) == ncclFloat16 ? kernel<half> : \
     (type) == ncclFloat32 ? kernel<float> : \
     (type) == ncclFloat64 ? kernel<double> : \
     RS_BF16_CASE(kernel, type) \
     RS_FP8_CASE(kernel, type) \
     nullptr) \
  )
#else
template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return testNotImplemented;
}
template <typename F>
testResult_t testLaunchDeviceKernelThreshold(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride) {
  return testNotImplemented;
}
template <typename F, typename H>
testResult_t testLaunchDeviceKernelThresholdLL(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, H llHandle) {
  return testNotImplemented;
}
template <typename F, typename H>
testResult_t testLaunchDeviceKernelThresholdLLFlag(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, H llHandle, int flag) {
  return testNotImplemented;
}
template <typename F, typename H, typename R>
testResult_t testLaunchDeviceKernelThresholdLLCtasFlag(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, H llHandle, int gridCtas, int flag, R flagBuf) {
  return testNotImplemented;
}
template <typename F, typename H>
testResult_t testLaunchDeviceKernelThresholdScratch(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, H scratchHandle) {
  return testNotImplemented;
}
template <typename F, typename H>
testResult_t testLaunchDeviceKernelThresholdScratchCtas(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, H scratchHandle, int gridCtas, size_t oneShotThresholdOverride) {
  return testNotImplemented;
}
template <typename F, typename H>
testResult_t testLaunchDeviceKernelAR2SplitCtas(F rsKernel, F agKernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, size_t sdmaThresholdOverride, H scratchHandle, int gridCtas) {
  return testNotImplemented;
}
#define SPECIALIZE_KERNEL(kernel, type, op) nullptr
#define SPECIALIZE_REDUCE_KERNEL(kernel, type, op) nullptr
#endif

typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncAllReduceWithBias = 5,
  ncclFuncSendRecv = 6,
  ncclFuncSend = 7,
  ncclFuncRecv = 8,
  ncclFuncAllToAllPivot = 9,
  ncclNumFuncs = 10
} ncclFunc_t;

typedef ncclResult_t (*rcclTestsGetAlgoInfo_t)(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                                          int collNetSupport, int nvlsSupport, int numPipeOps,
                                          int* algo, int* protocol, int* maxChannels);
typedef ncclResult_t (*rcclTestsGetAlgoName_t)(int algo, const char** algoName);
typedef ncclResult_t (*rcclTestsGetProtocolName_t)(int protocol, const char** protocolName);
typedef ncclResult_t (*rcclTestsGetSymkInfo_t)(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType, ncclRedOp_t op,
    int* algo, int* protocol, int* maxChannels);

extern rcclTestsGetAlgoInfo_t rcclTestsGetAlgoInfo;
extern rcclTestsGetProtocolName_t rcclTestsGetProtocolName;
extern rcclTestsGetAlgoName_t rcclTestsGetAlgoName;
extern rcclTestsGetSymkInfo_t rcclTestsGetSymkInfo;

// Network counter collector (self-contained, see collector.h for full API)
#include "collector.h"

// rccl-tests wrappers that bridge threadArgs ↔ collector API
extern NetworkCounterContext NetCounterCollectBefore(struct threadArgs* args);
extern void NetCounterCollectAfterAndPrint(struct threadArgs* args, const NetworkCounterContext& ctx);

#endif
