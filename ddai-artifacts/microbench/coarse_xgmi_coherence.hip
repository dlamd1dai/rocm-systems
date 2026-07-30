// coarse_xgmi_coherence.hip  (v2)
//
// Phase-2 (b) de-risk for the SDMA-scatter ReduceScatter idea. Zero RCCL/GIN
// dependency. Two questions, tested across the memory types that matter:
//
//   SPEED   : local CU read bandwidth of each buffer type (does the reduce stay
//             negligible vs the ~410 GB/s SDMA scatter?), including the COHERENT
//             read strategies (a coherent-but-slow read kills the design).
//
//   COHERENCE: faithfully mirrors GIN. A remote GPU (producer) writes a buffer
//             that physically lives on the consumer GPU over xGMI, then sets a
//             flag with a SYSTEM-scope release (== anvil signalPeer). The consumer
//             spins on the flag with a SYSTEM-scope ACQUIRE load (== gin.waitSignal)
//             then reads the data. We prime the consumer's L2 with the previous
//             epoch first to create a stale-line hazard, then count stale reads.
//
// Buffer types:
//   coarse    : hipMalloc                             (coarse-grained / cacheable)
//   finegrain : hipExtMallocWithFlags(Finegrained)    (fine-grained coherent)
//   vmmUncached: hipMemCreate(hipMemAllocationTypeUncached) + map  (== GIN window type)
//   vmmPinned  : hipMemCreate(hipMemAllocationTypePinned)   + map  (== #else GIN type)
//
// Read strategies: plain, nontemporal, per-elt system-acquire atomic.
//
// Build: hipcc -O3 --offload-arch=gfx950 coarse_xgmi_coherence.hip -o cxc
// Run:   ./cxc [prodDev consDev]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define HIPCHECK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ \
  printf("HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e_)); return 2;} } while(0)
#define HIPWARN(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ \
  printf("HIP warn %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e_)); } } while(0)

// 0-2: per-load strategies; 3-4: one-shot fence (agent/system acquire) then plain reads.
enum Strat { S_PLAIN=0, S_NT=1, S_ATOMIC=2, S_AGENTINV=3, S_SYSINV=4, S_NSTRAT=5 };
static const char* stratName(int s){
  switch(s){case S_PLAIN:return "plain      ";
            case S_NT:   return "nontemporal";
            case S_ATOMIC:return"sys-atomic ";
            case S_AGENTINV:return"agentfence+plain";
            case S_SYSINV:return "sysfence+plain ";}
  return "?";
}

__global__ void producer(uint32_t* data, unsigned long long* flag, size_t n, unsigned long long epoch){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=stride) data[i]=(uint32_t)epoch;
  __threadfence_system();
  if(blockIdx.x==0 && threadIdx.x==0) __hip_atomic_store(flag, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void prime(const uint32_t* data, size_t n, unsigned long long* sink){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x; unsigned long long a=0;
  for(; i<n; i+=stride) a+=data[i];
  if(a==0xdeadbeefULL) sink[0]=a;
}
__device__ __forceinline__ uint32_t rd(const uint32_t* p, int strat){
  if(strat==S_NT) return __builtin_nontemporal_load(p);
  if(strat==S_ATOMIC) return __hip_atomic_load(const_cast<uint32_t*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return *p;  // S_PLAIN, S_AGENTINV, S_SYSINV all use plain bulk loads (fence done once, below)
}
// one-shot cache-coherence op for the fence-then-plain strategies (emits L2 invalidate on CDNA)
__device__ __forceinline__ void oneShotAcquire(int strat){
  if(strat==S_AGENTINV) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  else if(strat==S_SYSINV) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // system scope
}
__global__ void consumer(const uint32_t* data, unsigned long long* flag, size_t n,
                         unsigned long long epoch, int strat, unsigned long long* mism){
  __shared__ int go;
  if(threadIdx.x==0){ while(__hip_atomic_load(flag,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM)<epoch){} go=1; }
  __syncthreads();
  oneShotAcquire(strat);  // one invalidate/acquire fence per thread before bulk plain reads
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x; unsigned long long bad=0;
  for(; i<n; i+=stride) if(rd(&data[i],strat)!=(uint32_t)epoch) bad++;
  if(bad) atomicAdd(mism,bad);
}
// SDMA-producer helpers: fill source on A, then hipMemcpyPeerAsync into B, then signal.
__global__ void fillBuf(uint32_t* s, size_t n, unsigned long long e){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=st) s[i]=(uint32_t)e;
}
__global__ void setFlag(unsigned long long* flag, unsigned long long e){
  __threadfence_system();
  __hip_atomic_store(flag, e, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void bwRead(const uint32_t* data, size_t n, int strat, unsigned long long* sink){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x; unsigned long long a=0;
  for(; i<n; i+=stride) a+=rd(&data[i],strat);
  if(a==0xdeadbeefULL) sink[0]=a;
}

// ---- VMM allocation with a chosen allocation type, mapped for both consumer(dev) and producer(peer) ----
static int allocVMM(void** out, size_t bytes, int dev, int peer, hipMemAllocationType type, size_t* mappedSz){
  hipMemAllocationProp prop{}; prop.type=type; prop.location.type=hipMemLocationTypeDevice; prop.location.id=dev;
  size_t gran=0; if(hipMemGetAllocationGranularity(&gran,&prop,hipMemAllocationGranularityMinimum)!=hipSuccess||gran==0) return 1;
  size_t sz=((bytes+gran-1)/gran)*gran; *mappedSz=sz;
  hipMemGenericAllocationHandle_t h;
  if(hipMemCreate(&h,sz,&prop,0)!=hipSuccess) return 1;
  void* p=nullptr;
  if(hipMemAddressReserve(&p,sz,0,0,0)!=hipSuccess){ hipMemRelease(h); return 1; }
  if(hipMemMap(p,sz,0,h,0)!=hipSuccess){ hipMemRelease(h); return 1; }
  hipMemAccessDesc acc[2];
  acc[0].location.type=hipMemLocationTypeDevice; acc[0].location.id=dev;  acc[0].flags=hipMemAccessFlagsProtReadWrite;
  acc[1].location.type=hipMemLocationTypeDevice; acc[1].location.id=peer; acc[1].flags=hipMemAccessFlagsProtReadWrite;
  if(hipMemSetAccess(p,sz,acc,2)!=hipSuccess){ hipMemUnmap(p,sz); hipMemRelease(h); return 1; }
  hipMemRelease(h);  // mapping keeps it alive
  *out=p; return 0;
}

static double bwGBs(size_t bytes,float ms){ return (double)bytes/(ms*1e-3)/1e9; }

// measure local read bandwidth on device `dev` for buffer `buf` under strategy
static double measureBW(void* buf, size_t nElts, int strat, int dev, int sms){
  size_t bytes=nElts*4; unsigned long long* sink=nullptr; hipMalloc(&sink,8);
  int blocks=0,thr=256; hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,bwRead,thr,0);
  int grid=blocks*sms; if(grid<=0) grid=2048;
  hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for(int w=0;w<3;w++) hipLaunchKernelGGL(bwRead,dim3(grid),dim3(thr),0,0,(uint32_t*)buf,nElts,strat,sink);
  hipDeviceSynchronize(); hipEventRecord(e0);
  const int R=20; for(int r=0;r<R;r++) hipLaunchKernelGGL(bwRead,dim3(grid),dim3(thr),0,0,(uint32_t*)buf,nElts,strat,sink);
  hipEventRecord(e1); hipEventSynchronize(e1); float ms=0; hipEventElapsedTime(&ms,e0,e1);
  hipEventDestroy(e0); hipEventDestroy(e1); hipFree(sink);
  return bwGBs(bytes*(size_t)R,ms);
}

// run coherence test: producer on A writes B's buffer, consumer on B reads under strat
static unsigned long long measureCoh(void* data,unsigned long long* flag,unsigned long long* mism,unsigned long long* sink,
                                     size_t n,int A,int B,hipStream_t sA,hipStream_t sB,int strat,int epochs){
  size_t bytes=n*4;
  hipSetDevice(B); hipMemset(flag,0,8); hipMemset(mism,0,8); hipMemset(data,0xFF,bytes); hipDeviceSynchronize();
  for(unsigned long long e=1;e<=(unsigned long long)epochs;e++){
    hipSetDevice(B); hipLaunchKernelGGL(prime,dim3(1024),dim3(256),0,sB,(uint32_t*)data,n,sink); hipStreamSynchronize(sB);
    hipSetDevice(A); hipLaunchKernelGGL(producer,dim3(1024),dim3(256),0,sA,(uint32_t*)data,flag,n,e);
    hipSetDevice(B); hipLaunchKernelGGL(consumer,dim3(1),dim3(1024),0,sB,(uint32_t*)data,flag,n,e,strat,mism);
    hipStreamSynchronize(sA); hipStreamSynchronize(sB);
  }
  unsigned long long m=0; hipMemcpy(&m,mism,8,hipMemcpyDeviceToHost); return m;
}

// SDMA producer: copy-engine writes B's buffer (A->B), then flag; consumer reads under strat.
static unsigned long long measureCohSDMA(void* dataB,uint32_t* srcA,unsigned long long* flag,unsigned long long* mism,
                                         unsigned long long* sink,size_t n,int A,int B,hipStream_t sA,hipStream_t sB,
                                         int strat,int epochs){
  size_t bytes=n*4;
  hipSetDevice(B); hipMemset(flag,0,8); hipMemset(mism,0,8); hipMemset(dataB,0xFF,bytes); hipDeviceSynchronize();
  for(unsigned long long e=1;e<=(unsigned long long)epochs;e++){
    hipSetDevice(B); hipLaunchKernelGGL(prime,dim3(1024),dim3(256),0,sB,(uint32_t*)dataB,n,sink); hipStreamSynchronize(sB);
    hipSetDevice(A);
    hipLaunchKernelGGL(fillBuf,dim3(1024),dim3(256),0,sA,srcA,n,e);
    hipMemcpyPeerAsync(dataB,B,srcA,A,bytes,sA);                 // <-- copy engine (SDMA) write A->B
    hipLaunchKernelGGL(setFlag,dim3(1),dim3(1),0,sA,flag,e);     // stream-ordered after the copy
    hipSetDevice(B); hipLaunchKernelGGL(consumer,dim3(1),dim3(1024),0,sB,(uint32_t*)dataB,flag,n,e,strat,mism);
    hipStreamSynchronize(sA); hipStreamSynchronize(sB);
  }
  unsigned long long m=0; hipMemcpy(&m,mism,8,hipMemcpyDeviceToHost); return m;
}

int main(int argc,char**argv){
  int A=0,B=1; if(argc>=3){A=atoi(argv[1]);B=atoi(argv[2]);}
  int nd=0; HIPCHECK(hipGetDeviceCount(&nd));
  printf("devices=%d producer=GPU%d consumer=GPU%d\n",nd,A,B);
  if(nd<2){printf("need >=2 GPUs\n");return 1;}
  int can=0; HIPCHECK(hipDeviceCanAccessPeer(&can,A,B)); printf("canAccessPeer(A->B)=%d\n",can);
  if(!can){printf("no P2P\n");return 1;}
  hipSetDevice(A); hipDeviceEnablePeerAccess(B,0);
  hipSetDevice(B); hipDeviceEnablePeerAccess(A,0);
  hipDeviceProp_t pB; HIPCHECK(hipGetDeviceProperties(&pB,B)); printf("consumer arch: %s  SMs=%d\n",pB.gcnArchName,pB.multiProcessorCount);

  const size_t nBW=(size_t)256*1024*1024/4;   // 256 MiB speed buffer
  const size_t nCoh=(size_t)64*1024*1024/4;    // 64 MiB coherence buffer
  const int EP=64;

  // ---- allocate all buffer variants on consumer B ----
  hipSetDevice(B);
  uint32_t *coarse=nullptr,*fine=nullptr; void *vmmU=nullptr,*vmmP=nullptr; size_t szU=0,szP=0;
  HIPCHECK(hipMalloc(&coarse,nBW*4));
  hipError_t fe=hipExtMallocWithFlags((void**)&fine,nBW*4,hipDeviceMallocFinegrained);
  int okU=allocVMM(&vmmU,nBW*4,B,A,hipMemAllocationTypeUncached,&szU)==0;
  int okP=allocVMM(&vmmP,nBW*4,B,A,hipMemAllocationTypePinned,&szP)==0;
  printf("alloc: coarse=ok fine=%s vmmUncached=%s vmmPinned=%s\n",
         fe==hipSuccess?"ok":"FAIL", okU?"ok":"FAIL", okP?"ok":"FAIL");

  // ================= SPEED =================
  printf("\n===== SPEED (local CU read, 256 MiB, GB/s) =====\n");
  printf("  %-16s %10s %12s %11s\n","buffer","plain","nontemporal","sys-atomic");
  auto row=[&](const char* nm, void* b){ if(!b){printf("  %-16s  (n/a)\n",nm);return;}
    HIPWARN(hipMemset(b,1,nBW*4)); HIPWARN(hipDeviceSynchronize());
    double p=measureBW(b,nBW,S_PLAIN,B,pB.multiProcessorCount);
    double t=measureBW(b,nBW,S_NT,B,pB.multiProcessorCount);
    double a=measureBW(b,nBW,S_ATOMIC,B,pB.multiProcessorCount);
    printf("  %-16s %10.1f %12.1f %11.1f\n",nm,p,t,a); return; };
  hipSetDevice(B);
  row("coarse",coarse);
  row("finegrain",fe==hipSuccess?fine:nullptr);
  row("vmmUncached",okU?vmmU:nullptr);
  row("vmmPinned",okP?vmmP:nullptr);

  // ================= COHERENCE =================
  hipSetDevice(A); hipStream_t sA; HIPCHECK(hipStreamCreate(&sA));
  hipSetDevice(B); hipStream_t sB; HIPCHECK(hipStreamCreate(&sB));
  unsigned long long *flag=nullptr,*mism=nullptr,*sink=nullptr;
  HIPCHECK(hipMalloc(&flag,8)); HIPCHECK(hipMalloc(&mism,8)); HIPCHECK(hipMalloc(&sink,8));
  printf("\n===== COHERENCE (GPU%d writes GPU%d's buffer over xGMI, 64 MiB, %d epochs; stale-elem count) =====\n",A,B,EP);
  printf("  %-16s %10s %10s %10s %14s %14s\n","buffer","plain","nontemp","sys-atomic","agentfence","sysfence");
  auto crow=[&](const char* nm, void* b){ if(!b){printf("  %-16s  (n/a)\n",nm);return;}
    unsigned long long p=measureCoh(b,flag,mism,sink,nCoh,A,B,sA,sB,S_PLAIN,EP);
    unsigned long long t=measureCoh(b,flag,mism,sink,nCoh,A,B,sA,sB,S_NT,EP);
    unsigned long long a=measureCoh(b,flag,mism,sink,nCoh,A,B,sA,sB,S_ATOMIC,EP);
    unsigned long long g=measureCoh(b,flag,mism,sink,nCoh,A,B,sA,sB,S_AGENTINV,EP);
    unsigned long long y=measureCoh(b,flag,mism,sink,nCoh,A,B,sA,sB,S_SYSINV,EP);
    printf("  %-16s %10llu %10llu %10llu %14llu %14llu\n",nm,p,t,a,g,y); return; };
  crow("coarse",coarse);
  crow("finegrain",fe==hipSuccess?fine:nullptr);
  crow("vmmUncached",okU?vmmU:nullptr);
  crow("vmmPinned",okP?vmmP:nullptr);

  // ===== COHERENCE with a real SDMA producer (copy engine), the faithful test =====
  hipSetDevice(A); uint32_t* srcA=nullptr; HIPCHECK(hipMalloc(&srcA,nCoh*4));
  printf("\n===== COHERENCE, SDMA producer (hipMemcpyPeerAsync A->B), 64 MiB, %d epochs; stale-elem count =====\n",EP);
  printf("  %-16s %10s %14s %10s\n","buffer","plain","agentfence","sys-atomic");
  auto srow=[&](const char* nm, void* b){ if(!b){printf("  %-16s  (n/a)\n",nm);return;}
    unsigned long long p=measureCohSDMA(b,srcA,flag,mism,sink,nCoh,A,B,sA,sB,S_PLAIN,EP);
    unsigned long long g=measureCohSDMA(b,srcA,flag,mism,sink,nCoh,A,B,sA,sB,S_AGENTINV,EP);
    unsigned long long a=measureCohSDMA(b,srcA,flag,mism,sink,nCoh,A,B,sA,sB,S_ATOMIC,EP);
    printf("  %-16s %10llu %14llu %10llu\n",nm,p,g,a); return; };
  srow("coarse",coarse);
  srow("vmmPinned",okP?vmmP:nullptr);
  hipFree(srcA);

  printf("\nMICROBENCH_DONE\n");
  return 0;
}
