# Conversation: GIN-GDA vs GIN-ROCSHMEM-API backend design

> **Note (2026):** The **GIN–rocSHMEM–API** backend (`NCCL_GIN_TYPE=4`) has been **removed** from RCCL. Use **GIN Anvil SDMA** (`NCCL_GIN_TYPE=6`) for intra-node xGMI device puts, or **GIN–rocSHMEM–GDA** (`NCCL_GIN_TYPE=5`) for IB GDA QPs. This document is retained for historical comparison only.

*Saved from chat, 2026-06-20.*

---

## User message 1

Read and understand GIN-GDA backend design and GIN-ROCSHMEM-API backend design. Explain the key differences in detail.

---

## Assistant reply (summary of sources)

Sources: RCCL in-tree docs (`docs/gin-anvil-sdma-backend-tests.md`), built-in plugins `projects/rccl/src/gin/gin_plugin_rocshmem_gda.cc`, and device templates `projects/rccl/src/include/nccl_device/gin/rocshmem_gda/gin_rocshmem_gda.h`.

### What they share

Both are **GIN net-device plugins**: they expose `ncclNetProperties_v12_t::netDeviceType` as a GIN variant, implement the same **host vtable** (`init`, `connect`, `createContext`, `regMrSym`, …), and expose a **device handle** so kernels can call the generic **`gin.put`** path. Neither sets async host `iput` / `iflush` vtable entries (device-side only). Both report `needsProxyProgress = 0`.

### 1. Role in the stack (`NCCL_GIN_TYPE` and naming)

| Aspect | **GIN–ROCSHMEM–API** (type **4**) | **GIN–ROCSHMEM–GDA** (type **5**) |
|--------|-----------------------------------|-----------------------------------|
| Constant | `NCCL_NET_DEVICE_GIN_ROCSHMEM_API` | `NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA` |
| Plugin name / `props->name` | `"rocshmem"` | `"rocshmem-gda"` |
| Doc summary | Device puts on top of **rocSHMEM’s public API**; **`ROCSHMEM_SDMA_ENABLED`** steers **SDMA policy** inside rocSHMEM (not GDA QPs). | Device puts on **GDA `QueuePair`s** built by RCCL’s **GIN GDA factory** (NIC → IB/DV → QP lifecycle); datapath is **`put_nbi`** on the QP. **GDA** lives under `rocshmem/src/gda/`, distinct from Anvil SDMA. |

High-level split: see `docs/gin-anvil-sdma-backend-tests.md` (section 2).

### 2. Host init and when the plugin “accepts” load

- **API**: `init()` only checks **`NCCL_GIN_TYPE == 4`**. It allocates `ginRocshmemInitCtx` and does **not** run a GDA hardware probe in that path.
- **GDA**: `init()` requires **type 5** and **`rocshmem_gin_probe_devices() > 0`**. If probe finds nothing, init fails—so GDA is tied to **discovered GIN/GDA-capable devices**.

### 3. `connect()` — where the network object is born

- **API**: `connect()` only fills **`nranks`, `rank`, `comm`**. No queues, no bootstrap exchange for QPs.
- **GDA**: `connect()` calls **`rocshmem_gin_create_qps(...)`** with **`bootstrapAllGather`** on the communicator’s bootstrap, fills **`rocshmem_gin_qp_set_t`** and **`gpu_qp_ptrs`**, then **`rocshmem_gin_init_constmem(...)`** for device-side constants.

### 4. Memory registration — symmetric heap vs explicit MR/rkey

- **API** (`regMrSym`): Resolves an **LSA flat address** via **`ncclDevrGetLsaSelfAddr`**, then on first use **`rocshmem::rocshmem_buffer_register_vmm(...)`** with the comm’s **`devrState`** (LSA base/size/`bigSize`). The device handle carries **`baseAddr`** (that flat address). Refcounting avoids duplicate `buffer_register` for the same pointer.
- **GDA** (`regMrSym`): Registers with the **QP set’s protection domain**: **`rocshmem_gin_reg_mr_vmm`**, gets **lkey/rkey**, **bootstrap allgathers** per-rank **rkeys and base VAs**, copies them to **GPU arrays** (`rkeys_dev`, `remote_vas_dev`). Device sees **`ncclGinRocshmemGdaMemHandle`** (local VA, lkey, device pointers to rkeys and remote VAs).

### 5. `createContext()` — what the device kernel actually holds

- **API**: Allocates **`ncclGinRocshmemGPUContext`** on device; **signals** from **`rocshmem::rocshmem_malloc`**; **counters** from `hipMalloc`. **No `QueuePair*` array** in the context.
- **GDA**: Context includes **`qps = (QueuePair**)cctx->gpu_qp_ptrs`** from `connect`. **Signals** use **`hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)`**, then **`rocshmem_gin_reg_mr`** on the signal buffer and another **allgather** of **signal rkeys and remote addresses** into device arrays **`signal_rkeys` / `signal_raddrs`**. Counters similarly use fine-grained device memory.

### 6. Device datapath (`gin.put`)

- **API**: Uses **`rocshmem::rocshmem_putmem(dst, src, bytes, peer)`** for data. Ordering uses **`rocshmem_fence`** (signal-only) vs **`rocshmem_quiet`** (when counters or stricter ordering). Signals use **`rocshmem_uint64_atomic_add`** on the **local signal pointer** (symmetric address model). **PutValue** uses **`rocshmem_*_p`** scalars after `quiet` when signaling.

- **GDA**: Cooperative group leader loads **`QueuePair* qp`** for the **peer**, then **`qp->put_nbi(...)`** with **explicit remote VA + rkey** and **local lkey**, optional **`qp->atomic_add`** for signals, **`qp->quiet`** when needed for counters, and **device threadfence** when scopes require it. **PutValue** uses **`put_nbi` with `lkey=0`** and inline source value.

### 7. rocSHMEM init ordering (tests / apps)

For **`NCCL_GIN_TYPE==4`**, rccl-tests registers **`rocshmemPreInit`** so **`rocshmem_init_attr` runs before `ncclCommInit`**. GDA mode uses the **standalone GDA QP factory** and is **less dependent** on full `rocshmem_init` heap semantics for the core datapath (shared global state can still matter in containers).

### 8. Operations / tuning that apply to one side only

- **`ROCSHMEM_SDMA_ENABLED`**: Documented as affecting **rocSHMEM’s SDMA path** for **type 4**, not the **GDA QueuePair** datapath for **type 5**.
- **Hardware gates**: GDA depends on **GDA provider + firmware** (e.g. BNXT minimum firmware in test harness); **`rocshmem_gin_probe_devices`** in `init` enforces that.

### Short takeaway

**GIN–ROCSHMEM–API** wires GIN to **rocSHMEM’s high-level GPU API** (`putmem`, fence/quiet, atomics) over **VMM/symmetric registration**; transport stays inside rocSHMEM (including **SDMA** when enabled).

**GIN–ROCSHMEM–GDA** wires GIN to **rocSHMEM’s GDA layer**: **GPU `QueuePair`s** at **`connect`**, **explicit MR/rkey/VA exchange**, and device code using **`put_nbi` / `atomic_add` / `quiet`** on those QPs—a **lower-level, QP-explicit** backend with **hardware probe** and **provider/firmware** constraints, separate from the **Anvil SDMA** module the **`ROCSHMEM_SDMA_ENABLED`** knob mainly concerns for type 4.

---

## User message 2

save the converation to a ".md" file.

---

## This file

Written to: `rocm-systems.git/docs/gin-gda-vs-gin-rocshmem-api-backend-design.md`.
