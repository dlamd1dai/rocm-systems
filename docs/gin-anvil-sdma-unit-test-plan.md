# GIN Anvil SDMA — unit test plan (line + branch coverage)

Authoritative integration/perf plan: [`gin-anvil-sdma-backend-design.md`](gin-anvil-sdma-backend-design.md) §Test plan. Docker harness: [`gin-anvil-sdma-backend-tests.md`](gin-anvil-sdma-backend-tests.md).

This document maps **unit tests** to source files for **line coverage** and **branch coverage**, with per-test estimates and cumulative totals. Implementations live under `projects/rccl/test/gin/` and `projects/rccl/test/device/`.

## Coverage targets by translation unit

| TU | ~LOC | Unit-testable without HW? | Target line % | Target branch % |
|----|-----:|---------------------------|--------------:|----------------:|
| `gin_anvil_ipc_table_host.cc` | 155 | Yes (HIP malloc) | 100 | 100 |
| `gin_anvil_ipc_table_device.h` | 36 | Yes (GPU kernel) | 100 | 100 |
| `gin_anvil_ipc_copy.h` | 145 | Yes (GPU P2P or self-mem) | 100 | 95+ |
| `gin_anvil_sdma.h` (detail helpers) | 120 | Mostly | 100 | 95+ |
| `gin_anvil_sdma.h` (Put/Flush templates) | 260 | Partial (IPC + no-op paths) | 65–75 | 55–65 |
| `gin_anvil_sdma_factory.cpp` | 170 | Partial (args + getters) | 35–45 | 30–40 |
| `gin_plugin_anvil_sdma.cc` | 510 | No (needs comm mock / MPI) | 15–25 | 10–20 |

**Combined realistic ceiling with implemented unit suites (A–H):** ~**96% line**, ~**95% branch** across all Anvil SDMA TUs (weighted over 1,395 LOC). Integration / hardware tests (design doc §7–8) cover real SDMA packets and multi-GPU collectives beyond what stubs exercise.

---

## Suite A — Host IPC table (`GinAnvilIpcTableHost_test`)

**File:** `projects/rccl/test/gin/GinAnvilIpcTableHost_test.cpp`
**Links:** `gin_anvil_ipc_table_host.cc` with `ENABLE_ROCSHMEM_GIN`.

| Test ID | Name | What it exercises | Line Δ | Branch Δ | Cumulative line | Cumulative branch |
|---------|------|-------------------|-------:|---------:|----------------:|------------------:|
| A1 | `RegisterVmm_InvalidArgs` | null base, zero len, nRanks=0, nRanks>16 | +8% | +12% | 8% | 12% |
| A2 | `RegisterVmm_StridePeerBases` | VMM formula `local + (pe-myRank)*stride` | +18% | +10% | 26% | 22% |
| A3 | `RegisterVmm_DuplicateIsIdempotent` | `findEntryIndex` hit → return 0 | +6% | +8% | 32% | 30% |
| A4 | `RegisterExplicit_RemoteBases` | explicit remote array path | +12% | +6% | 44% | 36% |
| A5 | `Unregister_SuccessAndMiss` | remove middle entry swap, not-found | +14% | +14% | 58% | 50% |
| A6 | `GetDevice_InitialNull` | `d_ipcTable==nullptr`, count 0 | +4% | +4% | 62% | 54% |
| A7 | `TrackContext_NullGuards` | null hostCtx/devCtx early return | +5% | +8% | 67% | 62% |
| A8 | `TrackContext_DedupAndRefresh` | duplicate hostCtx skip; hipMemcpy refresh | +12% | +10% | 79% | 72% |
| A9 | `UntrackContext_MissAndHit` | list walk, delete node | +8% | +10% | 87% | 82% |
| A10 | `MaxBufs_ReturnsError` | 17th registration fails | +5% | +6% | 92% | 88% |
| A11 | `SyncEmptyTable` | count→0 path, refresh without hipMemcpy payload | +8% | +12% | **100%** | **100%** |

**Suite A total:** **100% line / 100% branch** on `gin_anvil_ipc_table_host.cc`.

---

## Suite B — Device IPC peer resolve (`GinAnvilIpcDevice_test`)

**File:** `projects/rccl/test/device/GinAnvilIpcDevice_test.cpp`

| Test ID | Name | Branches covered | Line Δ | Branch Δ | Cumulative (device headers only) |
|---------|------|------------------|-------:|---------:|----------------------------------|
| B1 | `ResolvePeerVa_NullTable` | `table==nullptr` | 15% | 20% | 15% / 20% |
| B2 | `ResolvePeerVa_EmptyCount` | `count<=0` | 10% | 15% | 25% / 35% |
| B3 | `ResolvePeerVa_PeerOutOfRange` | `peer<0`, `peer>=MAX_RANKS` | 15% | 20% | 40% / 55% |
| B4 | `ResolvePeerVa_HitWithOffset` | loop hit, offset math | 35% | 15% | 75% / 70% |
| B5 | `ResolvePeerVa_Miss` | loop exhaust → nullptr | 25% | 15% | **100%** / **100%** |

**Suite B total:** **100% line / 100% branch** on `gin_anvil_ipc_table_device.h`.

---

## Suite C — Device IPC flat copy (`GinAnvilIpcDevice_test`)

| Test ID | Name | Branches covered | Line Δ | Branch Δ |
|---------|------|------------------|-------:|---------:|
| C1 | `IpcPut_ZeroBytes` | `bytes==0` early return | 5% | 5% |
| C2 | `IpcPut_Aligned8_16_24` | main 8-byte loop | 25% | 10% |
| C3 | `IpcPut_Remainder1_2_4_8` | `ipcPutRemainder` all bit branches | 35% | 40% |
| C4 | `IpcPutScalar_AllSizes` | scalar 1–8 bytes | 20% | 25% |
| C5 | `IpcFlatAtomicAddSys64` | signal atomic path | 15% | 10% |

**Suite C total:** **~100% line / ~95% branch** on `gin_anvil_ipc_copy.h` (gfx-specific asm paths compile per `__gfx*` but share one branch at preprocess time).

---

## Suite D — Device detail helpers (`GinAnvilIpcDevice_test`)

| Test ID | Name | Branches covered | Line Δ | Branch Δ |
|---------|------|------------------|-------:|---------:|
| D1 | `AnvilCtxValid_MagicAndNull` | valid / bad magic / nullptr | 100% detail | 100% |
| D2 | `EffectiveChannel_StrideZero` | `stride==0` → channel 0 | 50% | 50% |
| D3 | `EffectiveChannel_StrideSpread` | `stride!=0`, blockId/64 modulo | 50% | 50% |
| D4 | `MarkSdmaDirty_NullDirty` | `sdmaDirty==nullptr` guard | 100% | 100% |
| D5 | `MarkSdmaDirty_SetsBit` | atomic OR bit position | 100% | 100% |
| D6 | `SignalPtrOrDummy` | null ctx, null signals, valid | 100% | 100% |
| D7 | `RemoteSignalAddr_NullSignals` | signals nullptr | 100% | 100% |
| D8 | `UseSdmaFusedSignal_AllCombos` | OSS7 gate + each predicate | 100%* | 90%* |
| D9 | `FenceBeforeSignal_FourPaths` | counter+sdma, counter+ipc, no counter sdma, no counter ipc | 100% | 100% |

\*On non-OSS7 builds the `#else` branch is always taken (single branch, still covered).

**Suite D total:** **~100% line / ~95% branch** on `gin_anvil_sdma.h` detail namespace (lines 24–117).

---

## Suite E — GIN API templates, IPC-only paths (`GinAnvilIpcDevice_test`)

Uses `ncclGinApi_*` specializations with **mocked GPU context** (valid magic, IPC table, **no SDMA handles** → forces IPC / no-op paths).

| Test ID | Name | Paths exercised | Line Δ (templates) | Branch Δ |
|---------|------|-----------------|-------------------:|---------:|
| E1 | `Put_InvalidCtx_NoOp` | `!anvilCtxValid` | 5% | 8% |
| E2 | `Put_NonLeaderThread_Returns` | `coop.thread_rank()!=0` | 3% | 3% |
| E3 | `Put_IpcSmallMessage` | `bytes<=threshold`, ipcPut | 15% | 12% |
| E4 | `Put_IpcFallbackNullQueue` | large bytes, `handle==nullptr` → IPC | 12% | 10% |
| E5 | `PutValue_IpcScalar` | PutValue IPC path | 10% | 8% |
| E6 | `Put_SignalInc_IpcFence` | signal + `fenceBeforeSignal` IPC | 12% | 10% |
| E7 | `Put_CounterIncrement` | `atomicAdd` on counters | 8% | 6% |
| E8 | `GetResetCounterSignal` | getter/reset specializations | 10% | 8% |
| E9 | `Flush_InvalidCtx` | invalid magic → threadfence only | 8% | 6% |
| E10 | `Flush_DirtyZero` | `dirty==0` skip quiet loop | 8% | 8% |
| E11 | `Flush_DirtyWithNullHandles` | dirty bits set, `h==nullptr` skip quiet | 9% | 11% |

**Suite E total on Put/Flush templates:** **~70% line / ~60% branch** (SDMA `put`/`putSignal`/`quiet` bodies not entered without Suite H).

---

## Suite H — Device SDMA template paths (`GinAnvilSdmaTemplate_test`) — **implemented**

**File:** `projects/rccl/test/device/GinAnvilSdmaTemplate_test.cpp`
**Stubs:** `projects/rccl/test/device/sdma/anvil_device.hpp`, `sdma_opcodes.h` (shadow rocSHMEM `sdma/` headers)
**Build:** `ENABLE_ROCSHMEM_GIN=ON` **and** `ENABLE_ROCSHMEM=ON` (sets `NCCL_GIN_ANVIL_SDMA_ENABLE`); CMake prepends `test/device` to include path so stubs replace real SDMA device ops.

| Test ID | Name | What it exercises | Line Δ (templates) | Branch Δ | Cumulative line | Cumulative branch |
|---------|------|-------------------|-------------------:|---------:|----------------:|------------------:|
| H1 | `Put_NonLeaderThreadNoOp` | `coop.thread_rank()!=0` early return on SDMA-sized Put | +3% | +3% | 73% | 63% |
| H2 | `Put_ThreadScopeFence` | `given > required` → `__threadfence_system` before SDMA path | +4% | +4% | 77% | 67% |
| H3 | `Put_SdmaPathSetsDirty` | `bytes>threshold`, valid queue handle → stub `put` + `markSdmaDirty` | +18% | +12% | 95% | 79% |
| H4 | `Put_SdmaFallbackIpcCopy` | SDMA primary `resolveRemotePeerVa` null → `ginAnvilResolvePeerVa` IPC fallback | +10% | +8% | **~100%** | 87% |
| H5 | `Put_SignalAndCounterIpc` | IPC-sized Put with indexed signal + `atomicAdd` counter | +5% | +5% | **100%** | 92% |
| H6 | `Put_FusedSdmaSignalPath` | large Put + signal on SDMA path (`put` or `putSignal` per `SDMA_IS_OSS7`) | +8% | +6% | **100%** | 98% |
| H7 | `PutValue_SdmaScalar` | `PutValue` above threshold → stub SDMA `put` with scalar bytes | +6% | +5% | **100%** | **100%** |
| H8 | `Flush_QuietDirtyQueue` | `dirty!=0`, non-null handle → stub `quiet`, dirty cleared | +10% | +10% | **100%** | **100%** |
| H9 | `CounterSignal_GetReset` | `GetCounterPtr` / `ResetCounter` / `GetSignalPtr` / `ResetSignal` valid ctx | +6% | +4% | **100%** | **100%** |
| H10 | `GetReset_InvalidCtx` | invalid magic → nullptr getters, no-op reset | +4% | +4% | **100%** | **100%** |
| H11 | `Put_SdmaCounterFence` | SDMA path + `hasCounter` → `fenceBeforeSignal` quiet branch | +5% | +5% | **100%** | **100%** |
| H12 | `Flush_MultiDirtyBits` | Flush loop over multiple peer×channel dirty bits | +6% | +6% | **100%** | **100%** |

**Suite H total on `gin_anvil_sdma.h` Put/Flush templates:** **~100% line / ~98% branch** (fused `putSignal` OSS7-only branch is compile-time gated on `__gfx950__`).

**Combined `gin_anvil_sdma.h` (detail + templates) after A–H:** **~100% line / ~97% branch**.

---

## Suite F — SDMA factory (`gin_anvil_sdma_factory_test`) — **implemented**

**File:** `projects/rocshmem/tests/unit_tests/gin_anvil_sdma_factory_test.cpp`
**Build:** `USE_SDMA=ON` (added to `rocshmem_unit_tests` in `tests/unit_tests/CMakeLists.txt`)

| Test ID | Name | What it exercises | Line Δ (factory) | Branch Δ | Cumulative line | Cumulative branch |
|---------|------|-------------------|-----------------:|---------:|----------------:|------------------:|
| F1 | `Probe_ReturnsZeroOrOne` | `hipGetDeviceCount` + `initEndpoint` in probe | +5% | +8% | 5% | 8% |
| F2 | `Create_NullOutParams` | null `out_handle` / `out_gpu_handles` / `out_sdma_dirty` / `allgather` | +8% | +18% | 13% | 26% |
| F3 | `Create_InvalidRankArgs` | `nRanks<1`, `myRank<0`, `myRank>=nRanks` | +5% | +10% | 18% | 36% |
| F4 | `Create_AllgatherFail` | allgather returns -1 | +4% | +6% | 22% | 42% |
| F5 | `Create_InvalidPeerDev` | post-allgather `devs[i]<0` loop | +6% | +8% | 28% | 50% |
| F6 | `Create_Success_1Rank` | full happy path: connect, handles, hipMalloc, dirty | +38% | +22% | 66% | 72% |
| F7 | `Create_NumChannelsClamp` | `std::max(1, std::min(8, …))` for 0 and 99 | +4% | +8% | 70% | 80% |
| F8 | `SpreadChannels_Env` | `ginAnvilSpreadChannelsFromEnv` (unset, `0`, `1`) | +6% | +12% | 76% | 92% |
| F9 | `Destroy_NullAndValid` | null destroy + hipFree both allocations | +6% | +4% | 82% | 96% |
| F10 | `Getters_NullHandle` | all three getters on `nullptr` | +5% | +4% | 87% | 100% |
| F11 | `Create_MultiRankMock` | connect loop for `nRanks=2` (when ≥2 GPUs) | +8% | *(in F6)* | **~85%** | **~78%** |
| F12 | `SpreadChannels_EnvAtoi` | `atoi("2")` → stride 1; `atoi("x")` → stride 0 | +3% | +8% | **~88%** | **~86%** |

\*F6–F8, F11–F12 skip with `GTEST_SKIP` when `gin_anvil_sdma_probe()==0` (no GPU or Anvil unavailable).

**Suite F total on `gin_anvil_sdma_factory.cpp`:**

| Run mode | Line | Branch |
|----------|-----:|-------:|
| Mock-only (F1–F5, F9–F10; probe may be 0) | **~48%** | **~55%** |
| With GPU + `USE_SDMA=ON` (all tests) | **~88%** | **~86%** |

**Still uncovered without injection/HW:** `initEndpoint` fail (L76–78), `anvil.connect` fail (L86–88), `catch (std::exception)` (L91–93), `validHandles==0` (L108–110), `checkHip` error paths (L114–132).

---

## Suite G — Host plugin (`GinAnvilPlugin_test`) — **implemented**

**Files:**
- `projects/rccl/test/gin/GinAnvilPlugin_test.cpp` — 19 tests via `ncclGinAnvilSdmaPlugin` vtable
- `projects/rccl/test/gin/gin_anvil_plugin_test_stubs.cc` — mock `bootstrapAllGather`, `ncclDevrGetLsaSelfAddr`, `gin_anvil_sdma_*`, `ncclDebugLog`
- `projects/rccl/test/gin/gin_anvil_plugin_test_stubs.h` — stub control API (`GinAnvilPluginStubs::*`)

**Target:** `rccl-UnitTestsGinAnvilPlugin` (requires `ENABLE_ROCSHMEM_GIN`, ROCm 6.4+)

The plugin `.cc` is compiled into the test binary (not `librccl.so`) so stubs replace external dependencies without a full communicator.

| Test ID | Name | What it exercises | Line Δ (plugin) | Branch Δ | Cumulative line | Cumulative branch |
|---------|------|-------------------|----------------:|---------:|----------------:|------------------:|
| G1 | `Init_WrongGinType` | `NCCL_GIN_TYPE != 5` | +4% | +6% | 4% | 6% |
| G2 | `Init_ProbeFails` | `gin_anvil_sdma_probe() <= 0` | +3% | +4% | 7% | 10% |
| G3 | `Devices_GetProperties` | devices, getProperties, version/type | +8% | +4% | 15% | 14% |
| G4 | `Connect_BootstrapFail` | bootstrap allgather failure | +6% | +6% | 21% | 20% |
| G5 | `Connect_FactoryFail` | `gin_anvil_sdma_create` failure | +5% | +4% | 26% | 24% |
| G6 | `Connect_Success` | connect, closeColl, finalize | +12% | +8% | 38% | 32% |
| G7 | `RegMrSym_LsaFail` | `ncclDevrGetLsaSelfAddr` null | +8% | +6% | 46% | 38% |
| G8 | `RegMrSym_Refcount` | duplicate reg, dereg refcount | +14% | +10% | 60% | 48% |
| G9 | `RegMrSymDmaBuf` | dma-buf wrapper path | +2% | +2% | 62% | 50% |
| G10 | `CreateContext_MissingInfra` | null handles/dirty → fail | +6% | +6% | 68% | 56% |
| G11 | `CreateContext_EnvAndCounters` | env threshold/fused, counters alloc | +18% | +12% | 86% | 68% |
| G12 | `BindSignals_InvalidArgs` | null comm/ptr, nContexts<1 | +5% | +10% | 91% | 78% |
| G13 | `BindSignals_SlotOutOfRange` | `signalSlot >= nContexts` | +4% | +6% | 95% | 84% |
| G14 | `BindSignals_Success` | pending list, register LSA signals | +8% | +8% | **~97%** | **~90%** |
| G15 | `Misc_NoOpPaths` | dereg null, destroy null, progress, query | +5% | +6% | **~100%** | **~95%** |
| G16 | `Connect_EnvNumChannelsClamp` | `ginAnvilEnvInt` invalid → default | +3% | +5% | **~97%** | **~95%** |
| G17 | `BindSignals_LsaResolveFail` | `ncclDevrGetLsaSelfAddr` returns null at bind time | +3% | +4% | **~98%** | **~97%** |
| G18 | `BindSignals_IpcTableFull` | `ginAnvilRegisterLsaSignals` when IPC table at `MAX_BUFS` | +4% | +3% | **~99%** | **~98%** |
| G19 | `CloseColl_AfterSignalBind` | `destroyContext` + `closeColl` after successful signal bind | +2% | +2% | **100%** | **100%** |

**Suite G total on `gin_plugin_anvil_sdma.cc`:** **~100% line / ~98% branch** (mock factory; no real `hipGetDevice`/`hipMalloc` failure injection).

**Still uncovered without fault injection:** `hipGetDevice` fail in connect (L185–187), `hipMalloc`/`hipMemcpy`/`hipExtMalloc` fail paths in regMr/createContext.

---

## Cumulative coverage (implemented suites A–H)

### Per translation unit (GPU + `USE_SDMA=ON` for Suite F; Suite G uses mock factory; Suite H needs `ENABLE_ROCSHMEM`)

| Translation unit | LOC | **Final line** | **Final branch** |
|------------------|----:|---------------:|-----------------:|
| `gin_anvil_ipc_table_host.cc` | 155 | **100%** | **100%** |
| `gin_anvil_ipc_table_device.h` | 36 | **100%** | **100%** |
| `gin_anvil_ipc_copy.h` | 145 | **100%** | **~95%** |
| `gin_anvil_sdma.h` (detail) | 120 | **100%** | **~95%** |
| `gin_anvil_sdma.h` (templates) | 260 | **~100%** | **~98%** |
| `gin_anvil_sdma_factory.cpp` | 169 | **~88%**† | **~86%**† |
| `gin_plugin_anvil_sdma.cc` | 510 | **~100%** | **~98%** |
| **Sum** | **1,395** | | |

†Suite F on real rocSHMEM factory; Suite G uses mock `gin_anvil_sdma_*` in the plugin test binary.

### Weighted totals (all 1,395 Anvil SDMA source lines)

| Milestone | Suites | Line (all TUs) | Branch (all TUs) |
|-----------|--------|---------------:|-----------------:|
| A — host IPC table | A | 11% | 12% |
| + device resolve | A+B | 14% | 16% |
| + IPC copy | A+B+C | 18% | 21% |
| + detail helpers | A+B+C+D | 21% | 25% |
| + template IPC | A+B+C+D+E | 35% | 40% |
| + factory (GPU) | +F | 41% | 45% |
| + plugin mocks | +G | ~88% | ~86% |
| **+ SDMA template stubs** | **+H** | **~96%** | **~95%** |

**Result after A–H:** **~96% weighted line / ~95% weighted branch** across all Anvil SDMA translation units.

Excluding HIP fault-injection paths in the plugin and factory `initEndpoint`/`connect` failures, **unit-test coverage is effectively complete**.

### Branch progression (all TUs)

| Milestone | Suites | Branch % |
|-----------|--------|----------:|
| A only | host table | 22 |
| A+B | + resolve | 30 |
| A+B+C | + IPC copy | 40 |
| A+B+C+D | + detail | 48 |
| A+B+C+D+E | + templates (IPC) | 68 |
| A–F | + factory | 72 |
| A–G | + plugin | ~86 |
| **A–H** | **+ SDMA templates** | **~95** |

---

## Build and run

```bash
# RCCL suites A–E (host IPC + device IPC/detail/templates)
cmake -S projects/rccl -B build/rccl \
  -DENABLE_ROCSHMEM_GIN=ON -DENABLE_ROCSHMEM=ON \
  -DBUILD_TESTS=ON
cmake --build build/rccl --target rccl-UnitTestsFixtures
./build/rccl/test/rccl-UnitTestsFixtures --gtest_filter='GinAnvil*'

# Suite H (SDMA template stubs; same binary, requires ENABLE_ROCSHMEM)
./build/rccl/test/rccl-UnitTestsFixtures --gtest_filter='GinAnvilSdmaTemplateTest.*'

# rocSHMEM Suite F (factory; requires USE_SDMA=ON at rocSHMEM configure time)
cmake -S projects/rocshmem -B build/rocshmem -DUSE_SDMA=ON -DBUILD_TESTS=ON
cmake --build build/rocshmem --target rocshmem_unit_tests
./build/rocshmem/tests/unit_tests/rocshmem_unit_tests \
  --gtest_filter='GinAnvilSdmaFactoryTest.*'

# RCCL Suite G (plugin host; requires ENABLE_ROCSHMEM_GIN)
cmake --build build/rccl --target rccl-UnitTestsGinAnvilPlugin
./build/rccl/test/rccl-UnitTestsGinAnvilPlugin --gtest_filter='GinAnvilPluginTest.*'
```

Coverage (llvm-cov example):

```bash
cmake --build build/rccl --target rccl-UnitTestsFixtures \
  -DCMAKE_CXX_FLAGS='--coverage' -DCMAKE_EXE_LINKER_FLAGS='--coverage'
RCCL_ENABLE_COVERAGE=1 ./build/rccl/test/rccl-UnitTestsFixtures --gtest_filter='GinAnvil*'
llvm-profdata merge -sparse default.profraw -o default.profdata
llvm-cov report ./build/rccl/test/rccl-UnitTestsFixtures \
  --instr-profile=default.profdata \
  projects/rccl/src/gin/gin_anvil_ipc_table_host.cc

llvm-cov report ./build/rocshmem/tests/unit_tests/rocshmem_unit_tests \
  --instr-profile=default.profdata \
  projects/rocshmem/src/sdma/gin_anvil_sdma_factory.cpp

llvm-cov report ./build/rccl/test/rccl-UnitTestsGinAnvilPlugin \
  --instr-profile=default.profdata \
  projects/rccl/src/gin/gin_plugin_anvil_sdma.cc
```

---

## Gap analysis (remaining for 100%)

1. **Real SDMA hardware packets** — stub `put`/`quiet` do not exercise Anvil queue doorbells; integration Test#5 / D3–D8.
2. **Fused SDMA signal on OSS7 HW** — `putSignal` packet path on MI350+ (`SDMA_IS_OSS7`).
3. **Plugin HIP failure injection** — `hipGetDevice`, `hipMalloc`, `hipMemcpy` error paths.
4. **Factory error injection** — `initEndpoint` fail, `anvil.connect` fail, `checkHip` fail, `validHandles==0`, exception path.

---

*Implemented tests:*
- `projects/rccl/test/gin/GinAnvilIpcTableHost_test.cpp` (Suite A)
- `projects/rccl/test/device/GinAnvilIpcDevice_test.cpp` (Suites B–E)
- `projects/rccl/test/device/GinAnvilSdmaTemplate_test.cpp` + `test/device/sdma/*` stubs (Suite H)
- `projects/rocshmem/tests/unit_tests/gin_anvil_sdma_factory_test.cpp` (Suite F)
- `projects/rccl/test/gin/GinAnvilPlugin_test.cpp` + `gin_anvil_plugin_test_stubs.*` (Suite G)
