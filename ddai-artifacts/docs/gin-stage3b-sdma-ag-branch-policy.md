# GIN SDMA AllGather — Branch Policy

**Scope:** `users/dondai/gin-stage3b-sdma-ag*`, fork PR #4, MI355 validation
**End state:** Upstream AG on modern `develop`; MI355 runtime stays on NCCL 2.30.7 via `-nccl-2.30.7-wip`

---

## Branch roles

| Branch | Status | Purpose |
|---|---|---|
| **`users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip`** | **Active — MI355 canonical** | Long-lived NCCL 2.30.7 / docker validation line (`rccl-gin-gda-sdma-*`). All new MI355 AG work lands here. |
| **`users/dondai/gin-stage3b-sdma-ag`** | **Active — upstream only** | Clean PR branch for [fork PR #4](https://github.com/dlamd1dai/rocm-systems/pull/4). Rebased on current `develop`. No `ddai-artifacts/` churn. |
| **`users/dondai/gin-stage3b-sdma-ag-wip`** | **Frozen / archive** | Historical staging branch. Do not commit new work. Superseded by `-nccl-2.30.7-wip`. |

---

## Where to commit

| Change type | Target branch |
|---|---|
| AllGather kernel, policy, devtime, rccl-tests AG logic for **MI355** | `-nccl-2.30.7-wip` |
| Same change intended for **upstream review** | `-nccl-2.30.7-wip` first → cherry-pick/rebase to `gin-stage3b-sdma-ag` |
| Docker harness, perf scripts, MI355 logs, manifest sync checks | `-nccl-2.30.7-wip` only |
| PR cleanup (drop artifacts, rebase onto `develop`) | `gin-stage3b-sdma-ag` only |
| Upstream-only harness changes (modern rccl-tests / `os.h`) | `gin-stage3b-sdma-ag` → after merge, backport if needed |

**Rule:** Never develop directly on `-wip` or on the SUT working tree without pushing back to `-nccl-2.30.7-wip`.

---

## Sync directions

```
gin-stage3b-sdma-ag-wip  (frozen)
         │
         ▼
gin-stage3b-sdma-ag-nccl-2.30.7-wip  ◄── MI355 SUT, docker images
         │         ▲
         │         │ selective cherry-pick / backport
         ▼         │
gin-stage3b-sdma-ag  ──►  ROCm/develop  (via PR #4, then future PRs)
```

| Direction | When | How |
|---|---|---|
| **NCCL wip → PR branch** | Preparing or updating upstream PR | Cherry-pick AG commits; rebase `gin-stage3b-sdma-ag` onto `develop`; drop `ddai-artifacts/` |
| **NCCL wip → SUT** | After every push to `-nccl-2.30.7-wip` | `git fetch && git reset --hard origin/...` on `smci355` |
| **develop → NCCL wip** | After upstream AG merge or related fixes (#10658, #10672, #10675, etc.) | Cherry-pick; resolve rccl-tests API drift manually |
| **NCCL wip → NCCL wip** | Normal development | Commit on branch; push `origin` |

**Do not:** rebase `-nccl-2.30.7-wip` onto modern `develop` wholesale.

---

## MI355 validation (required before push)

**SUT:** `dondai@smci355-ccs-aus-m03-17:~/rocm-systems/`
**Docker image:** `rccl-gin-gda-sdma-713` (or successor pinned to NCCL 2.30.7)

Minimum gate for AG changes:

1. Sync SUT to branch tip (`reset --hard origin/users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip`)
2. Rebuild `all_gather_perf` (and `alltoall_perf` if shared harness touched) in docker
3. Run AG smoke (hybrid `-D 3`, validation on)
4. Log under `ddai-artifacts/logs/` on SUT

---

## Pre-push checklist

- [ ] Commits on correct branch (`-nccl-2.30.7-wip` vs `gin-stage3b-sdma-ag`)
- [ ] No accidental `ddai-artifacts/` or perf logs on PR branch
- [ ] If upstream-bound: corresponding cherry-pick prepared for `gin-stage3b-sdma-ag`
- [ ] MI355 build + smoke passed (for `-nccl-2.30.7-wip`)
- [ ] SUT HEAD matches `origin` branch tip

---

## Quick verify

```bash
# Canonical MI355 branch tip
git fetch origin users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip
git rev-parse origin/users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip

# SUT in sync
ssh dondai@smci355-ccs-aus-m03-17 \
  'cd ~/rocm-systems && git rev-parse HEAD && git rev-parse origin/users/dondai/gin-stage3b-sdma-ag-nccl-2.30.7-wip'

# PR branch (upstream path)
git fetch origin users/dondai/gin-stage3b-sdma-ag
git log -1 --oneline origin/users/dondai/gin-stage3b-sdma-ag
```

---

## Related upstream stack (modern line)

These target `develop`, not the NCCL 2.30.7 line directly:

| PR | Branch | Relationship to AG |
|---|---|---|
| [#10658](https://github.com/ROCm/rocm-systems/pull/10658) | `gin-sdma-a2a-devtime` | A2A devtime — backport to `-nccl-2.30.7-wip` if needed |
| [#10672](https://github.com/ROCm/rocm-systems/pull/10672) | `gin-gda-a2a-unittests` | Unit tests — backport selectively |
| [#10675](https://github.com/ROCm/rocm-systems/pull/10675) | `gin-scope-fence-guard-fix` | Fence fix — backport if AG path hits same code |

After each lands on `develop`, evaluate **cherry-pick to `-nccl-2.30.7-wip`**, not automatic merge.
