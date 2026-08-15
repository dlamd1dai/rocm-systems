# Integration patch: fabric branch in `ginAnvilRegMrSym`

Target: `projects/rccl/src/gin/gin_plugin_anvil_sdma.cc`, function
`ginAnvilRegMrSym` (currently lines ~332-356 on `meta-455-wip`). NOT compiled;
review + build on the SUT.

Today the peer table is always stride-based:

```332:356:projects/rccl/src/gin/gin_plugin_anvil_sdma.cc
  const ptrdiff_t stride = (ptrdiff_t)devr->bigSize;
  uintptr_t* remote_vas_host = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)cctx->nranks);
  ...
  for (int pe = 0; pe < cctx->nranks; pe++) {
    remote_vas_host[pe] = (uintptr_t)lsaSelfAddr + static_cast<ptrdiff_t>(pe - devr->lsaSelf) * stride;
  }
  ...
  hostMh.baseAddr = (uintptr_t)lsaSelfAddr;
  hostMh.remote_vas = mh->remote_vas_dev;
  hostMh.vmmStride = stride;
```

## Proposed hybrid

Include `gin_anvil_sdma_fabric.h`. Build the peer table via one of two paths:

```cpp
  const ptrdiff_t stride = (ptrdiff_t)devr->bigSize;
  uintptr_t* remote_vas_host = (uintptr_t*)malloc(sizeof(uintptr_t) * (size_t)cctx->nranks);
  if (!remote_vas_host) { /* existing cleanup */ }

  ptrdiff_t hostStride = stride;  // tier-1 stride by default

#ifdef HIP_FABRIC_API
  GinAnvilFabricPeers* fab = nullptr;
  if (ncclGinAnvilUseFabricPath(cctx->comm)) {
    // (A) Get this buffer's cuMem allocation handle (NEW core hook).
    CUmemGenericAllocationHandle selfHandle{};
    size_t handleSize = 0;
    NCCLCHECK(ncclDevrGetMemHandle(devr, data, &selfHandle, &handleSize));

    // (B) Clique-wide fabric exchange -> per-peer imported VAs.
    fab = new (std::nothrow) GinAnvilFabricPeers(cctx->comm);
    if (fab == nullptr) { /* cleanup */ return ncclSystemError; }
    NCCLCHECK(fab->exchange(lsaSelfAddr, selfHandle, size));

    // (C) Fill the peer table. Intra-node peers can keep the LSA-stride VA (so
    // SDMA still works); cross-node peers use the fabric-imported VA (shader-core
    // ipcPut path). Simplest correct choice: use the fabric VA for ALL peers.
    for (int pe = 0; pe < cctx->nranks; pe++) {
      remote_vas_host[pe] = reinterpret_cast<uintptr_t>(fab->getPeerVa(pe));
    }
    hostStride = 0;  // force device tier-2 (remote_vas) resolution

    // Keep the pointer alive for teardown; stash on mh (add field mh->fabricPeers).
    mh->fabricPeers = fab;
  } else
#endif
  {
    for (int pe = 0; pe < cctx->nranks; pe++) {
      remote_vas_host[pe] = (uintptr_t)lsaSelfAddr + static_cast<ptrdiff_t>(pe - devr->lsaSelf) * stride;
    }
  }

  // ... existing hipMalloc + hipMemcpy of remote_vas_host -> mh->remote_vas_dev ...

  // Also register the explicit peer bases for the fallback/self-test tier + signals.
  (void)ncclGinAnvilIpcTableRegisterExplicit(lsaSelfAddr, remote_vas_host, cctx->nranks, size);

  ncclGinAnvilSdmaMemHandle hostMh;
  hostMh.baseAddr = (uintptr_t)lsaSelfAddr;
  hostMh.remote_vas = mh->remote_vas_dev;
  hostMh.vmmStride = hostStride;   // 0 on the fabric path
```

## `ginAnvilMemHandle` + teardown

Add a field:

```cpp
struct ginAnvilMemHandle {
  ...
#ifdef HIP_FABRIC_API
  GinAnvilFabricPeers* fabricPeers = nullptr;
#endif
};
```

In `ginAnvilDeregMrSym` (`gin_plugin_anvil_sdma.cc:368-388`) add before `free(mh)`:

```cpp
#ifdef HIP_FABRIC_API
  if (mh->fabricPeers) { delete mh->fabricPeers; mh->fabricPeers = nullptr; }
#endif
```

## New core hook (dev_runtime)

```cpp
// dev_runtime.cc: resolve the cuMem allocation handle backing `data`.
// Single-segment fast path; return ncclInvalidUsage for multi-segment buffers
// until the exchange supports per-segment fabric handles.
ncclResult_t ncclDevrGetMemHandle(struct ncclDevrState* devr, void* data,
                                  CUmemGenericAllocationHandle* outHandle, size_t* outSize) {
  // Look up the ncclDevrMemory for `data` (same lookup as ncclDevrGetLsaSelfAddr,
  // dev_runtime.cc:2480-2503), then return mem->memHandles[0] + its size.
}
```

## Notes / open items

- **Single-node** does not need this patch if the LSA heap itself is
  fabric-backed (design §2a); this branch is for **cross-node clique** peers.
- Choosing "fabric VA for ALL peers" (C) is simplest and correct; a later
  optimization keeps intra-node peers on the SDMA queue via their LSA-stride VA
  while cross-node peers use the fabric VA (mixed transport within one collective).
- Multi-segment registered buffers need per-segment fabric handles; scope the
  first cut to single-segment (assert otherwise).
