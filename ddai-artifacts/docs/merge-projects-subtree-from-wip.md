# Merging the `projects/` subtree from `-wip` into `gin-stage2c-sdma-bcast`

**Task:** bring the `projects/` changes from `users/dondai/gin-stage2c-sdma-bcast-wip`
(source) into `users/dondai/gin-stage2c-sdma-bcast` (target), without dragging in the
rest of the `-wip` branch.

**Date:** 2026-07-27

---

## Branch state (measured)

| Fact | Value |
|---|---|
| merge-base of the two branches | `d6d378e242` — **equals the target's HEAD** |
| Relationship | target is a **strict ancestor** of `-wip`; `-wip` = target + **685 commits** |
| `projects/` diff (target..wip) | **1143 files, +177,157 / −49,332** |
| ↳ `projects/rccl-tests` only (your GIN-SDMA work) | **11 files, +1,745 / −51**, in 11 clean commits |
| ↳ `projects/` **excluding** `rccl-tests` (upstream sync) | **1132 files, +175,412** (rocSHMEM, hsa-runtime, …) — *not* your work |

**Implication:** a literal "merge the whole `projects/` subtree" pulls in the ~175k-line
upstream sync too, not just the GIN-SDMA rccl-tests changes.

### The 11 self-contained `projects/rccl-tests` commits (oldest → newest)

All verified to touch **only** `projects/rccl-tests/` (no stray paths), so they
cherry-pick cleanly:

```
a62fc3dbce rccl-tests: GIN-SDMA hybrid broadcast (-D 3) for single-node xGMI
fcbb685f0f rccl-tests: split GIN-SDMA LSA<->GIN threshold per collective
afe1b771f9 rccl-tests: size-hybrid GIN-SDMA AlltoAll (-D 3) with env var threshold
9b7afed97e rccl-tests: optimize GIN-SDMA AllGather LSA branch for small/mid messages
b1748ea849 rccl-tests: add LL packed data+flag fast path to GIN-SDMA AllGather
cae1d62dc0 rccl-tests: add opt-in LL fast path to GIN-SDMA AllToAll (default off)
0b1e2d154b rccl-tests: add LL fast path to GIN-SDMA Broadcast for small messages
7fd5c10c4d rccl-tests: add scatter+allgather large-message tier to GIN-SDMA Broadcast
f0cea5ea4f rccl-tests: extract GIN-SDMA collective policy into a shared header
bd1eca2b41 rccl-tests: add GIN-SDMA collective policy unit and GPU functional tests
ca5e747db6 rccl-tests: add gcov coverage target for the GIN-SDMA policy tests
```

---

## Key fact about git

There is **no native 3-way merge limited to a path**. A normal `git merge` would pull in
all 685 commits (including the upstream sync). To scope to a subtree you either
**cherry-pick** the relevant commits (keeps history) or **path-checkout** the end-state
(one squashed commit).

---

## Option 1 — Recommended: cherry-pick just the rccl-tests work

Brings the GIN-SDMA broadcast/allgather/alltoall + policy tests, **preserves the 11
commit messages**, and pulls in **zero** upstream noise and **zero** `ddai-artifacts`
scratch:

```bash
git switch users/dondai/gin-stage2c-sdma-bcast
git cherry-pick $(git rev-list --reverse \
  users/dondai/gin-stage2c-sdma-bcast..users/dondai/gin-stage2c-sdma-bcast-wip \
  -- projects/rccl-tests)
```

`rev-list --reverse … -- projects/rccl-tests` yields exactly those 11 commits in
oldest→newest apply order. Since the target is their base and they only touch
`projects/rccl-tests`, they apply without conflict.

## Option 1b — Same scope, squashed to one commit

If you don't care about preserving the 11-commit history:

```bash
git switch users/dondai/gin-stage2c-sdma-bcast
git checkout users/dondai/gin-stage2c-sdma-bcast-wip -- projects/rccl-tests
git commit -m "rccl-tests: GIN-SDMA broadcast/allgather/alltoall + policy tests"
```

## Option 2 — Literal request: the entire `projects/` subtree

> **Warning:** this also brings the 1132-file / +175k-line upstream sync (rocSHMEM,
> hsa-runtime, …), not just the GIN-SDMA work. Only use it if you want the target's
> `projects/` to match `-wip`'s exactly.

```bash
git switch users/dondai/gin-stage2c-sdma-bcast
git rm -r --quiet projects/                                  # handles deletions for an exact match
git checkout users/dondai/gin-stage2c-sdma-bcast-wip -- projects/
git commit -m "Merge projects/ subtree from gin-stage2c-sdma-bcast-wip"
```

---

## After whichever option

```bash
git switch users/dondai/gin-stage2c-sdma-bcast-wip   # (optional) return to wip
# and when ready:
git push origin users/dondai/gin-stage2c-sdma-bcast
```

**Recommendation:** Option 1 — it matches the likely intent (promote the clean
rccl-tests work into the non-wip branch), keeps history, and avoids the accidental
175k-line upstream import.
