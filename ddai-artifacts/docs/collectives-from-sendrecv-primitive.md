# Composing collectives from the Send/Recv primitive (and how GIN-SDMA realizes a ring step)

This note sketches (1) how the general point-to-point `Send`/`Recv` primitive
composes into the standard collectives, and (2) how the GIN-SDMA device kernels
realize one "Send/Recv round" as `gin.put -> waitSignal -> flush`, including
where the reduction `+=` slots in for ReduceScatter.

It is background/design context for the GIN-SDMA collectives expansion (P1
movement collectives already landed; P2/P3 add the reduction collectives).

---

## Part A - Composing collectives from Send/Recv

At the bottom, every collective is a **choreography of point-to-point Send/Recv
over a virtual topology** - a ring, a tree, or a butterfly (hypercube). The
"algorithm" of a collective *is* the schedule: which peer each rank sends to and
receives from, in which step, and what local op it applies between steps.

### 0. The one rule that shapes everything: deadlock avoidance

A bare `Send` blocks until matched. So every step where a rank *both* sends and
receives must fuse the two so the whole rank-set issues at once:

```c
ncclGroupStart();
  ncclSend(sbuf, n, t, next, comm, stream);   // to my successor
  ncclRecv(rbuf, n, t, prev, comm, stream);   // from my predecessor
ncclGroupEnd();                                // both fire together -> ring can drain
```

That fused pair - the "sendrecv" - is the atom the ring algorithms are built
from. Trees and butterflies use the same fused pair, just with different peers
per step.

### 1. Ring AllGather

Rank `r` starts with chunk `r`; everyone ends with all `N` chunks. Lay ranks in a
ring (`next = r+1`, `prev = r-1`) and rotate `N-1` times, forwarding the chunk you
just received (store-and-forward):

```text
recvbuf[r] = mychunk                     // seed my own slot
cur = r
for step in 0 .. N-2:
    send_slot = cur
    recv_slot = (cur - 1 + N) % N
    group:
        Send(recvbuf[send_slot] -> next)
        Recv(recvbuf[recv_slot] <- prev)
    cur = recv_slot
// after N-1 steps every slot is filled
```

Bandwidth-optimal: each rank sends `(N-1)` chunks, `(N-1)/N * totalbytes` per
link. No reduction - pure movement (the "second half" of ring AllReduce).

### 2. Ring ReduceScatter

Rank `r` ends with the fully-reduced value of chunk `r`. Same ring, but reduce
each passing chunk in as it arrives:

```text
cur = r
for step in 0 .. N-2:
    send_slot = (cur) % N
    recv_slot = (cur - 1 + N) % N
    group:
        Send(accum[send_slot] -> next)       // pass along partial sum
        Recv(tmp <- prev)
    accum[recv_slot] += tmp                  // fold in the incoming partial
    cur = recv_slot
// accum[r] now holds the global reduction of chunk r
```

Same `N-1` steps and traffic as AllGather, plus one local `+=` (an `Apply<op,T>`)
per step. The "first half" of ring AllReduce.

### 3. Ring AllReduce = ReduceScatter then AllGather

```text
ReduceScatter()   // each rank owns the reduced chunk r
AllGather()       // broadcast every reduced chunk to everyone
```

Each phase moves `(N-1)/N` of the buffer per link, so AllReduce moves
`2*(N-1)/N ~= 2x` the data - the ring optimum. Every message is one fused
Send/Recv; the only compute is the `+=` folded into the ReduceScatter phase.

> This is why the P1 movement kernels matter for P2/P3: **AllGather == ReduceScatter
> without the `+=`**, and Gather/Scatter are the "collapse to a root" versions of
> the same shifts. The GIN-SDMA `gin.put(peer, ...)` is the device-initiated analog
> of `Send`; the receiver-side `waitSignal` is the analog of the matched `Recv`
> completing.

### 4. Broadcast and Reduce (chain / tree)

Pipelined ring (chain) broadcast - root injects, each rank forwards to `next`:

```text
if r == root: Send(buf -> next)
elif r == last: Recv(buf <- prev)
else: group { Recv(buf <- prev); Send(buf -> next) }   // relay
```

Split `buf` into slices and pipeline them to overlap links -> near
bandwidth-optimal. **Reduce** is the mirror image: leaves send, interior nodes
`Recv` from children, `+=`, then `Send` to parent; the root ends with the sum.
Over a binary tree this is `log N` steps (latency-optimal for small messages)
instead of the ring's `N-1` (bandwidth-optimal for large) - the same size-driven
algorithm choice as the LL-vs-LSA-vs-GIN tiering, one level up.

### 5. AllToAll - direct personalized exchange

No reduction, no forwarding; `N-1` distinct sends per rank, all fused:

```text
group:
    for p in 0 .. N-1:
        Send(sendbuf[p] -> p)      // my chunk destined for p
        Recv(recvbuf[p] <- p)      // p's chunk destined for me
// (p == r is a local copy)
```

For large `N` this is often reorganized into log-step Bruck or pairwise-exchange
schedules to cut the message count, but semantically it's all-pairs Send/Recv.

### 6. Recursive doubling / halving-doubling (butterfly)

For small messages the latency-optimal schedules pair ranks along hypercube
dimensions, doubling the distance each step (`log N` steps):

```text
// recursive doubling AllGather / AllReduce
for step in 0 .. log2(N)-1:
    partner = r XOR (1 << step)
    group:
        Send(mydata -> partner)
        Recv(theirs <- partner)
    mydata = combine(mydata, theirs)   // concat (AllGather) or reduce (AllReduce)
```

`log N` messages but each grows - good latency, worse bandwidth than ring. Real
libraries pick ring vs. recursive-doubling by message size (the tiering theme).

### The unifying picture

| Collective        | Topology  | Steps      | Per-step local op        | Send/Recv role              |
|-------------------|-----------|------------|--------------------------|-----------------------------|
| AllGather         | ring      | `N-1`      | store forwarded chunk    | fused pair, rotate          |
| ReduceScatter     | ring      | `N-1`      | `accum += incoming`      | fused pair, rotate          |
| AllReduce         | ring      | `2(N-1)`   | `+=` (first half only)   | RS then AG                  |
| Broadcast         | chain/tree| `N-1`/`logN`| none (relay)            | Recv-then-Send              |
| Reduce            | tree      | `log N`    | `+=` at interior nodes   | Recv(children)->Send(parent)|
| AllToAll          | full mesh | 1 group    | none                     | `N-1` distinct sends        |
| AllReduce (small) | butterfly | `log N`    | combine                  | XOR-partner exchange        |

Two invariants make it all work:

1. **Every collective decomposes into rounds of matched Send/Recv** over a chosen
   virtual topology; the topology + step schedule is the algorithm.
2. **The only extra ingredient beyond movement is the per-step local `combine`**
   (`+=` for sum, `max`, etc.). Pure-movement collectives (AllGather, Broadcast,
   Scatter, Gather, AllToAll, SendRecv) omit it; reduction collectives
   (ReduceScatter, AllReduce, Reduce) add it - precisely the `Apply<op,T>` functor
   + scratch window the P2/P3 plan introduces.

---

## Part B - How the GIN-SDMA device kernels realize a ring step

### 1. One GIN "Send/Recv round" = put -> waitSignal -> flush

The SendRecv GIN tier (`projects/rccl-tests/src/sendrecv.cu`,
`GinSendRecvKernel`) is a single ring step expressed in one-sided (push) form:

```cpp
const unsigned int signalIndex = 0;
ncclGin gin { devComm, /*context=*/0 };
const uint64_t signalValue = gin.readSignal(signalIndex);

ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

// One put to the send peer (issue once, by the single global thread 0).
if (tid == 0) {
  gin.put(ncclTeamWorld(devComm), sendPeer,
      recvwin, recvoffset,
      sendwin, sendoffset,
      msgBytes, ncclGin_SignalInc{signalIndex});
}
// Each rank receives exactly one put (from its recv peer).
gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + 1);
gin.flush(ncclCoopCta());
```

Correspondence to the abstract `group { Send(->next); Recv(<-prev) }`:

| Abstract op            | GIN device realization                                             | Meaning                                                                        |
|------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `readSignal`           | `signalValue = gin.readSignal(idx)`                                | snapshot my completion counter *before* the round (the base to wait against)   |
| entry barrier          | `bar.sync(...)`                                                    | all ranks' `recvbuf`s quiescent before anyone writes them (kills the initData memset race; the analog of "posting the Recv buffer") |
| `Send(-> sendPeer)`    | `gin.put(team, sendPeer, dstWin,dstOff, srcWin,srcOff, bytes, SignalInc{idx})` | **one-sided push**: reads *my* `sendbuf`, writes directly into the *peer's* `recvbuf`, atomically bumps the **peer's** signal by 1 |
| `Recv(<- recvPeer)` done | `gin.waitSignal(idx, signalValue + 1)`                           | block until *my* signal reaches base+1 - the one put destined for me has landed |
| memory ordering        | `gin.flush()`                                                     | drain/settle outstanding ops so received data is visible locally and the round retires |

Crucial structural point: **GIN is one-sided (put/RDMA), not two-sided.** There is
no separately posted `Recv`. The sender already knows the receiver's buffer address
(symmetric registered window / peer VA), so it writes the destination directly and
rings a doorbell (`SignalInc`). The "Recv" degenerates to a *counter wait*; the
receiver never names a source in `waitSignal`, it just counts arrivals.

### 2. The signal count *is* the collective's fan-in

Because `SignalInc` increments the **receiver's** counter once per incoming put,
"how many puts do I receive?" fully characterizes completion - and that number is
the topology's in-degree:

| Collective (GIN tier) | Puts each rank *issues*   | Puts each rank *receives* -> waits            |
|-----------------------|---------------------------|-----------------------------------------------|
| SendRecv (ring)       | 1 (to `r+1`)              | 1 (from `r-1`) -> `base+1`                     |
| Broadcast (flat)      | root: `N-1`; others: 0    | non-root: 1 -> `base+1`; root: 0 (flush only) |
| Gather                | non-root: 1 -> root       | root: `N-1` -> `base+(N-1)`; others: 0        |
| Scatter               | root: `N-1`; others: 0    | non-root: 1 -> `base+1`                        |
| AllGather / AllToAll  | `N-1` each                | `N-1` each -> `base+(N-1)`                      |

This is the asymmetry tuned in the policy header (`ginSignalCount` and the
per-collective wait values). Broadcast/Scatter roots *flush without waiting* (they
only send); Gather roots wait `base+(N-1)`; the ring waits `base+1`.

### 3. Where the `+=` slots in for ReduceScatter

In pure movement (SendRecv/AllGather) the put lands straight in the final
`recvbuf` and you're done. Reduction can't do that - the sender must **not**
blindly overwrite the receiver's accumulator, because the receiver needs to
*combine* incoming data with what it already holds. So one extra ingredient is
required: a **staging scratch window** on the receiver, plus a receiver-side
`Apply<op,T>` between the wait and the next forward. That is the
`resourceWindow`/scratch the P2 plan introduces.

A ring ReduceScatter step becomes:

```text
scratch = resourceWindow(chunkBytes)          // per-CTA staging, from ncclDevResourceRequirements
accumChunkIdx = r
for step in 0 .. N-2:
    base = gin.readSignal(idx)
    entry/step barrier                         // enforce step ordering (see below)

    // ---- Send: push my current partial into the PEER'S SCRATCH (not its final buf) ----
    if (tid == 0)
        gin.put(team, sendPeer,
                scratchWin@peer, scratchOff,   // <-- destination is peer's scratch
                accumWin, sendChunkOff,        // <-- source is my running accumulator
                chunkBytes, ncclGin_SignalInc{idx});

    // ---- Recv-complete: my incoming partial has landed in MY scratch ----
    gin.waitSignal(idx, base + 1);             // acquire: data-in-scratch now visible

    // ---- the composition step: fold it in (this is the "+=") ----
    Apply<op,T>(accum + recvChunkOff,          // dst: my accumulator slice
                scratch,                        // src: what just arrived
                chunkCount);                    // all CUs cooperate over the chunk

    gin.flush(ncclCoopCta());                   // release before it is forwarded next step
    accumChunkIdx = (accumChunkIdx - 1 + N) % N
// accum[r] now holds the global reduction of chunk r
```

Two things change versus the movement kernels:

1. **Destination redirection.** The put targets `scratchWin@peer`, not the peer's
   final buffer. Movement collectives put straight into `recvbuf`; reduction stages
   into scratch so the local accumulator survives until the combine. This is the
   entire reason the reduction collectives need the scratch requirement and the
   movement ones don't.
2. **The `Apply<op,T>` between `waitSignal` and the next `put`.** `waitSignal` is
   the *acquire* that guarantees the arrived partial is visible before the CUs read
   it; `flush` is the *release* that guarantees the folded result is settled before
   it ships onward. The reduction operator (`ncclSum`/`max`/...) is the only compute
   in the whole thing - everything else is the same put/signal movement skeleton.

AllReduce then reuses this verbatim and appends a movement AllGather phase (no
`Apply`, put straight into `recvbuf`) - literally the ReduceScatter kernel followed
by the AllGather kernel.

### 4. The one subtlety: step ordering / pipelining

Movement SendRecv is a *single* round, so a lone entry barrier suffices. A ring
reduce is `N-1` **dependent** rounds - step `k+1`'s combine must not run before
step `k`'s partial is folded and forwarded, or you reduce stale data. Two ways to
enforce it, mirroring the LSA-vs-LL barrier tradeoff:

- **Barrier per step** (simple, correct, higher latency) - a team barrier between
  rounds, like the LSA tier's entry/exit barriers.
- **Epoch/signal-encoded pipelining** (what LL does) - let the signal value carry
  the step number (`waitSignal(base + step + 1)`), so a rank proceeds as soon as
  *its* predecessor's step-`k` put arrives, without a global barrier. This is the
  same double-buffer-by-epoch trick as the LL A2A session - and it inherits the same
  "bounded skew" hazard reasoned about for SendRecv LL (a fast rank must not lap a
  slow reader), which is why SendRecv LL keeps one entry barrier.

### Through-line

`gin.put(...SignalInc)` + `waitSignal` + `flush` is one Send/Recv round; the
topology decides the peers and the signal-wait count; and the single added
`Apply<op,T>` over a scratch-staged partial is what upgrades a movement ring into a
reduction ring. P2/P3 is mostly "add the scratch window + the functor + step
ordering" on top of the movement skeleton P1 already built and validated.
