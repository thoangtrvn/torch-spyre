# Layout-Safe 2-D all_reduce Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SpyreCCLBackend::allreduce(SUM)` correct for 2-D `[tokens, hidden]` tensors at ≥4 ranks by composing it from the layout-safe pairwise `reduce_scatter` + `all_gather` primitives (never byte-slicing a single tiled buffer), un-parking the WIP that implements those primitives.

**Architecture:** For a ≥2-D input, `all_reduce = reduce_scatter(SUM) → all_gather` when `tokens ≥ world_size`, else `all_gather → local on-device sum`. Chunks are produced as whole per-rank tensors via torch-layer `narrow(0)+clone` (the `alltoall_base` pattern), so every transfer is a whole identically-shaped tensor whose `SpyreTensorLayout` tiling is reconstructed correctly on the receiver. The existing 1-D / uniform paths are untouched.

**Tech Stack:** C++17 (torch-spyre PrivateUse1 backend), spyre-comms (C++20), PyTorch c10d ProcessGroup, torchrun, Google Test / pytest.

## Global Constraints

- **Branch:** `phase-1-overlap-communicate-compute` (land with the closed async Phase-1 work, per user decision).
- **Controller/subagents CANNOT build C++.** All builds + HW runs are performed by the USER on a 4-Spyre pod (`tmhoangt-spyre-dev-bob-quick`). Every "run test" step is a **HW checkpoint**: implementer writes+syncs the code/script, then the controller asks the user to build and run, and waits for the pasted result before proceeding. Ask the user to rebuild+run at EVERY point HW testing becomes possible (do not batch to the end) — ref memory `feedback-prompt-hw-rebuild-test-checkpoints`.
- **Sync target:** `tmhoangt-spyre-dev-bob-quick:/home/tmhoangt/spyre-multi/macos/` (per `sync_to_sentient.sh` rsync pattern). The SSH host shares `$HOME` with the 4-card machine; controller may READ result files from the shared mount but the USER launches HW runs. `env.sh` lives only on the pod; source with `set +u` around it.
- **Un-parked WIP:** `spyre_ccl.{cpp,hpp}` (~429 lines) + spyre-comms `create_context`/`getSubComm`/wireup are currently UNCOMMITTED, byte-intact vs tag `sdd-task5-wip-backup`. Task 1 commits them. Until then, when staging `spyre_ccl.*` use hand-built `--cached` patches / `git add -p`, NEVER `git add <file>`.
- **Commits:** `git commit -s` (DCO). NO `Co-Authored-By` trailer. Pre-commit hooks fail on pre-existing unrelated mypy/clang-format debt and reformat the WIP file — commit with `--no-verify` after confirming YOUR staged files are format-clean.
- **Reduction op:** backend supports SUM only (existing `TORCH_CHECK`). Do not add other ops.
- **Layout rule (load-bearing):** NEVER slice a ≥2-D tiled tensor by `BufferDesc` byte range or device byte offset. Chunk at the torch layer via `.narrow(0, off, len)` then materialize with `.clone()` (own `SpyreTensorLayout`). `prepare_buffer_desc` ignores `storage_offset` — never use it for a sub-range descriptor. Ref memory `spyre-eager-tiled-layout-breaks-comms-slicing`.
- **No new async:** the composed path calls the existing SYNCHRONOUS reduce_scatter/allgather (they host-wait internally). Do not attempt round overlap (blocked on single `comm_stream_` + per-call `address_exchange_.invalidate()`).

---

## File Structure

| File | Responsibility | Task |
|------|----------------|------|
| `torch_spyre/csrc/distributed/spyre_ccl.cpp` | Un-park `reduce_scatter`/`exchange_uneven`/variable-`allgather`; add 2-D dispatch + compose helper in `allreduce` | 1, 2, 3 |
| `torch_spyre/csrc/distributed/spyre_ccl.hpp` | Un-park `exchange_uneven`/`error_store_key_` decls; add `allreduce_2d_compose` private decl | 1, 2 |
| `flex-opensource/spyre-comms/src/...` (subgroup/wireup) | Un-park `create_context`/`getSubComm` | 1 |
| `tests/distributed/test_allreduce_2d.py` (NEW) | Correctness regression: 2-D reduces fully at TP=2/4/8, uneven, decode | 4 |
| `run_allreduce_2d_verify.sh` (NEW, workspace root) | Plain-torchrun HW runner for the regression test (NOT pytest — avoids vLLM engine build) | 4 |
| `bench_allreduce_2d.sh` (NEW, workspace root) | Benchmark gate: p50/p95 latency prefill+decode at TP=4 | 5 |

---

## Task 1: Un-park the reduce_scatter / exchange_uneven / subgroup WIP

**Files:**
- Modify: `torch_spyre/csrc/distributed/spyre_ccl.cpp` (commit the parked `reduce_scatter`, `exchange_uneven`, variable-`allgather` hunks)
- Modify: `torch_spyre/csrc/distributed/spyre_ccl.hpp` (commit `exchange_uneven` + `error_store_key_` decls)
- Modify: `flex-opensource/spyre-comms/...` (commit the parked `create_context`/`getSubComm`/wireup)

**Interfaces:**
- Produces (for Task 2):
  - `c10::intrusive_ptr<Work> SpyreCCLBackend::reduce_scatter(std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors, const ReduceScatterOptions& opts)` — c10d list form: `inputTensors[0]` is a world-length list of per-rank chunk tensors; `outputTensors[0]` receives this rank's fully-reduced chunk. SUM only. Synchronous (host-waits).
  - `c10::intrusive_ptr<Work> SpyreCCLBackend::allgather(std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors, const AllgatherOptions&)` — list form; variable path via `exchange_uneven` already present.
  - `std::unique_ptr<spyre_comms::WorkSchedule> SpyreCCLBackend::exchange_uneven(const spyre_comms::BufferDesc& send_base, const spyre_comms::BufferDesc& recv_base, int peer, int round_tag)` — private helper.

- [ ] **Step 1: Verify the WIP is byte-intact vs the backup tag before committing**

```bash
cd $DTI_PROJECT_ROOT/torch-spyre
# Confirm the parked hunks are present and unchanged since parking.
git diff --stat -- torch_spyre/csrc/distributed/spyre_ccl.cpp torch_spyre/csrc/distributed/spyre_ccl.hpp
# Spot-check the three WIP entry points exist in the worktree:
grep -c "SpyreCCLBackend::reduce_scatter\|SpyreCCLBackend::exchange_uneven" torch_spyre/csrc/distributed/spyre_ccl.cpp
# Expect: reduce_scatter body (not the throw stub) + exchange_uneven definition present.
```
Expected: `reduce_scatter` has a real body (calls `exchange_uneven`, not `SpyreCCLNotSupportedException`); `exchange_uneven` defined; `.hpp` has `error_store_key_` + `exchange_uneven` decl.

- [ ] **Step 2: Stage ALL parked WIP hunks in spyre_ccl.{cpp,hpp} (whole-file is correct NOW — this task's deliverable IS the WIP)**

```bash
cd $DTI_PROJECT_ROOT/torch-spyre
# For THIS task only, the entire spyre_ccl.* diff IS the deliverable, so a
# whole-file add is correct (the "never git add spyre_ccl" rule protected the
# WIP from being bundled into OTHER commits; here the WIP is the commit).
git add torch_spyre/csrc/distributed/spyre_ccl.cpp torch_spyre/csrc/distributed/spyre_ccl.hpp
# Stage the spyre-comms subgroup/wireup WIP:
cd $DTI_PROJECT_ROOT/flex-opensource
git add spyre-comms/src   # scope to the create_context/getSubComm/wireup changes
git status --porcelain | grep -E "spyre_ccl|spyre-comms"
```

- [ ] **Step 3: Sync both repos to the pod and ask the user to build + HW-regress the world path**

Sync (controller):
```bash
cd $DTI_PROJECT_ROOT
rsync -aq torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.cpp torch-spyre/torch_spyre/csrc/distributed/spyre_ccl.hpp \
  tmhoangt-spyre-dev-bob-quick:/home/tmhoangt/spyre-multi/macos/torch-spyre/torch_spyre/csrc/distributed/
rsync -aq flex-opensource/spyre-comms/src/ \
  tmhoangt-spyre-dev-bob-quick:/home/tmhoangt/spyre-multi/macos/flex-opensource/spyre-comms/src/
```
Then ask the user (HW checkpoint): rebuild spyre-comms (flex build) AND torch-spyre (`uv sync --all-extras --active`), then run the EXISTING world-path regression to catch the `g0_` comm_id key change (ref memory `spyre-subgroup-allgatherv-parked-wip`):
```bash
cd /home/tmhoangt/spyre-multi
bash run_async_dispatch_verify.sh 4              # async suite still green
cd torch-spyre && uv run pytest tests/distributed/test_allgatherv.py -v -m "not upstream"  # variable allgather + uniform
```
Expected: async suite PASS (world path unaffected by comm_id change); allgatherv PASS. **If the `g0_` comm_id change regresses the world path, STOP and report — it is a blocker, not a Task-2 problem.**

- [ ] **Step 4: Add a reduce_scatter list-form HW test to prove the un-parked primitive works**

Create `tests/distributed/test_reduce_scatter.py` if not already present with a uniform 2-D case (rank k contributes chunk list; each rank gets the summed chunk). If the file exists (it is untracked per earlier `git status`), verify it covers 4-rank uniform reduce_scatter and reuse it.

```python
# test body (uniform reduce_scatter, 4 ranks): each rank supplies a world-length
# list of [rows, HIDDEN] chunks all filled with (rank+1); after reduce_scatter,
# this rank's output chunk == sum_k (k+1) == world*(world+1)/2, elementwise.
def test_reduce_scatter_uniform_2d(self):
    r, n = self.comm_rank, self.comm_size
    HIDDEN, ROWS = 4096, 8
    ins = [torch.full((ROWS, HIDDEN), float(r + 1), dtype=torch.float16, device=DEVICE)
           for _ in range(n)]
    out = torch.empty((ROWS, HIDDEN), dtype=torch.float16, device=DEVICE)
    dist.reduce_scatter(out, ins, op=dist.ReduceOp.SUM)
    got = out.to("cpu")
    expected = float(n * (n + 1) // 2)
    assert torch.equal(got, torch.full_like(got, expected)), \
        f"rank {r}: reduce_scatter got {got.flatten()[:4].tolist()} want {expected}"
```

- [ ] **Step 5: Ask the user to run the reduce_scatter test (HW checkpoint)**

Sync the test, then ask the user:
```bash
cd /home/tmhoangt/spyre-multi/torch-spyre/tests/distributed
torchrun --nproc-per-node 4 -m pytest test_reduce_scatter.py -v -m "not upstream"
```
Expected: PASS on all 4 ranks. This confirms the un-parked reduce_scatter primitive is correct in isolation before allreduce composes it.

- [ ] **Step 6: Commit (only after user confirms Steps 3 + 5 green)**

```bash
cd $DTI_PROJECT_ROOT/torch-spyre
git add torch_spyre/csrc/distributed/spyre_ccl.cpp torch_spyre/csrc/distributed/spyre_ccl.hpp tests/distributed/test_reduce_scatter.py
git commit -s --no-verify -m "feat(distributed): un-park layout-safe reduce_scatter + exchange_uneven

Activate the parked pairwise-sendrecv reduce_scatter (list form) and the
exchange_uneven credit-safe sub-leg decomposition, plus the spyre-comms
subgroup create_context/getSubComm/wireup they depend on. HW-verified:
world-path regression (async suite + allgatherv) green after the g0_
comm_id key change; uniform 2-D reduce_scatter reduces correctly at 4 ranks."
cd $DTI_PROJECT_ROOT/flex-opensource && git add spyre-comms/src && git commit -s --no-verify -m "feat(spyre-comms): un-park subgroup create_context/getSubComm/wireup"
```

---

## Task 2: allreduce ≥2-D compose path (tokens ≥ world_size)

**Files:**
- Modify: `torch_spyre/csrc/distributed/spyre_ccl.cpp:656-708` (`allreduce` — add ≥2-D dispatch + `allreduce_2d_compose` helper)
- Modify: `torch_spyre/csrc/distributed/spyre_ccl.hpp` (declare `allreduce_2d_compose`)

**Interfaces:**
- Consumes (from Task 1): `reduce_scatter(...)`, `allgather(...)` list forms.
- Produces (for Task 3): `c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce_2d_compose(at::Tensor& tensor, const AllreduceOptions& opts)` — private helper handling the ≥2-D case; Task 3 extends it with the `tokens < world` branch.

- [ ] **Step 1: Add the dispatch gate in `allreduce` (after the splittability guard, before `enqueue_async`)**

In `spyre_ccl.cpp::allreduce`, insert before the existing `seq_.fetch_add`/`enqueue_async` (line ~692):
```cpp
    // Layout-safe ≥2-D path: the flat-byte Ring reduce-scatter under-reduces a
    // stickified ≥2-D tensor (a linear byte chunk straddles sticks of
    // interleaved rows). Compose from reduce_scatter+all_gather over whole
    // per-rank chunk tensors instead. 1-D stays on the (correct) enqueue path.
    const int world_sz2 = static_cast<int>(group_context_->getSize());
    if (world_sz2 > 1 && tensors[0].dim() >= 2) {
      return allreduce_2d_compose(tensors[0], opts);
    }
```

- [ ] **Step 2: Declare the helper in `spyre_ccl.hpp`** (next to the other private collective helpers)

```cpp
  // Layout-safe all_reduce(SUM) for a ≥2-D tensor: reduce_scatter+all_gather
  // over whole per-rank chunk tensors (torch-layer narrow+clone chunking), so
  // no tiled buffer is ever byte-sliced. See
  // docs/superpowers/specs/2026-08-05-2d-allreduce-layout-safe-fix-design.md.
  c10::intrusive_ptr<c10d::Work> allreduce_2d_compose(
      at::Tensor& tensor, const c10d::AllreduceOptions& opts);
```

- [ ] **Step 3: Implement `allreduce_2d_compose` (tokens ≥ world path only; Task 3 adds the else branch)**

```cpp
c10::intrusive_ptr<Work> SpyreCCLBackend::allreduce_2d_compose(
    at::Tensor& tensor, const AllreduceOptions& opts) {
  const int world = static_cast<int>(group_context_->getSize());
  const int64_t tokens = tensor.size(0);

  // Task 3 replaces this guard with the tokens<world all_gather+sum fallback.
  TORCH_CHECK(tokens >= world,
              "[", getBackendName(), "]: allreduce_2d_compose tokens (", tokens,
              ") < world_size (", world, ") not yet handled (see Task 3).");

  // ── Chunk dim-0 into `world` whole tensors at the TORCH layer ──
  // narrow(0)+clone: each chunk is its own freshly-tiled tensor (never a
  // byte-range slice of the tiled input). Uneven tokens: first `rem` chunks get
  // one extra row (matches exchange_uneven's balanced+remainder decomposition).
  const int64_t base = tokens / world;
  const int64_t rem = tokens % world;
  std::vector<at::Tensor> chunks;
  chunks.reserve(world);
  int64_t off = 0;
  for (int i = 0; i < world; ++i) {
    const int64_t len = base + (i < rem ? 1 : 0);
    chunks.push_back(tensor.narrow(0, off, len).clone());  // own SpyreTensorLayout
    off += len;
  }
  const int me = static_cast<int>(group_context_->getRank());

  // ── reduce_scatter: this rank ends with chunks[me] fully reduced ──
  std::vector<at::Tensor> rs_out_list = {at::empty_like(chunks[me])};
  std::vector<std::vector<at::Tensor>> rs_in_list = {chunks};
  ReduceScatterOptions rs_opts;
  rs_opts.reduceOp = ReduceOp::SUM;
  reduce_scatter(rs_out_list, rs_in_list, rs_opts)->wait();

  // ── all_gather: every rank reassembles all `world` reduced chunks ──
  std::vector<at::Tensor> ag_slots;
  ag_slots.reserve(world);
  for (int i = 0; i < world; ++i) {
    const int64_t len = base + (i < rem ? 1 : 0);
    ag_slots.push_back(at::empty({len, tensor.size(1)}, tensor.options()));
  }
  std::vector<std::vector<at::Tensor>> ag_out_list = {ag_slots};
  std::vector<at::Tensor> ag_in_list = {rs_out_list[0]};
  allgather(ag_out_list, ag_in_list, AllgatherOptions{})->wait();

  // ── concat reduced chunks back into the caller's tensor (torch layer) ──
  off = 0;
  for (int i = 0; i < world; ++i) {
    const int64_t len = ag_slots[i].size(0);
    tensor.narrow(0, off, len).copy_(ag_slots[i]);
    off += len;
  }

  seq_.fetch_add(1, std::memory_order_relaxed);
  auto ws = group_context_->device_copy(prepare_buffer_desc(tensor),
                                        prepare_buffer_desc(tensor));
  ws->SetStreamAffinity(comm_stream_);
  ws->start();
  ws->wait();
  return c10::make_intrusive<SpyreCCLWork>(
      OpType::ALLREDUCE, std::move(ws),
      /*hold=*/std::vector<at::Tensor>{tensor},
      /*result=*/std::vector<at::Tensor>{tensor}, op_timeout_);
}
```
Note: the trailing `device_copy(tensor,tensor)` synthesizes a completed Work so the c10d Future/wait path is well-formed (mirrors the reduce_scatter degenerate branch). The real reduction already completed synchronously above.

- [ ] **Step 4: Sync + ask the user to build and run the 2-D verification (HW checkpoint — full test lands in Task 4; here just prove it reduces)**

Sync `spyre_ccl.{cpp,hpp}` (hand-built `--cached` NOT needed now — the WIP is committed as of Task 1, so a whole-file rsync of content is fine). Ask the user to rebuild torch-spyre and run the existing diagnostic (already on the pod), which now must show FULL:
```bash
cd /home/tmhoangt/spyre-multi
bash diag_allreduce_split_coverage.sh 4
```
Expected: `VERDICT` shows 2-D `[8,4096]` `FULL=yes` (`reduced=32768`) on all ranks — the bug is fixed for `tokens ≥ world`. 1-D still FULL.

- [ ] **Step 5: Commit (after user confirms FULL=yes)**

```bash
cd $DTI_PROJECT_ROOT/torch-spyre
git add torch_spyre/csrc/distributed/spyre_ccl.cpp torch_spyre/csrc/distributed/spyre_ccl.hpp
git commit -s --no-verify -m "fix(distributed): layout-safe 2-D all_reduce via reduce_scatter+all_gather

Compose all_reduce(SUM) for >=2-D [tokens,hidden] from the layout-safe
pairwise reduce_scatter + all_gather over whole per-rank chunk tensors
(torch-layer narrow+clone), instead of the flat-byte Ring reduce-scatter
that under-reduced a stickified >=2-D tensor. HW-verified: [8,4096] at 4
ranks now reduces fully (was 4096/32768). tokens<world handled in the
follow-up. Fixes the TP>=4 activation-allreduce correctness bug."
```

---

## Task 3: Decode fallback — tokens < world_size (all_gather + local sum)

**Files:**
- Modify: `torch_spyre/csrc/distributed/spyre_ccl.cpp` (`allreduce_2d_compose` — replace the `tokens >= world` TORCH_CHECK with the else branch)

**Interfaces:**
- Consumes: `allgather(...)` list form (Task 1).

- [ ] **Step 1: Replace the Task-2 guard with the tokens<world branch**

In `allreduce_2d_compose`, replace the `TORCH_CHECK(tokens >= world, ...)` with:
```cpp
  if (tokens < world) {
    // Chunking dim-0 across `world` ranks is impossible when tokens < world
    // (decode: tokens==1). Every rank all_gathers all `world` whole
    // [tokens,hidden] contributions (whole-tensor transfers are layout-safe)
    // and sums them on-device. Unconditionally correct for any tokens.
    std::vector<at::Tensor> slots;
    slots.reserve(world);
    for (int i = 0; i < world; ++i) {
      slots.push_back(at::empty_like(tensor));
    }
    std::vector<std::vector<at::Tensor>> ag_out = {slots};
    std::vector<at::Tensor> ag_in = {tensor.clone()};  // own layout, not a view
    allgather(ag_out, ag_in, AllgatherOptions{})->wait();

    tensor.copy_(slots[0]);
    for (int i = 1; i < world; ++i) {
      tensor.add_(slots[i]);  // on-device accumulate
    }
    order_after_caller_stream(tensor);

    seq_.fetch_add(1, std::memory_order_relaxed);
    auto ws = group_context_->device_copy(prepare_buffer_desc(tensor),
                                          prepare_buffer_desc(tensor));
    ws->SetStreamAffinity(comm_stream_);
    ws->start();
    ws->wait();
    return c10::make_intrusive<SpyreCCLWork>(
        OpType::ALLREDUCE, std::move(ws),
        std::vector<at::Tensor>{tensor}, std::vector<at::Tensor>{tensor},
        op_timeout_);
  }
  // else: tokens >= world -> reduce_scatter+all_gather (Task 2 code below)
```

- [ ] **Step 2: Sync + ask the user to run the decode case (HW checkpoint)**

Ask the user to rebuild torch-spyre and run a quick decode probe (add a `[1,4096]` case to the diag script's shape list, or run inline). Expected: `[1,4096]` at 4 ranks reduces fully (`reduced=4096`, every element == 4.0).

- [ ] **Step 3: Commit (after user confirms decode green)**

```bash
git add torch_spyre/csrc/distributed/spyre_ccl.cpp
git commit -s --no-verify -m "fix(distributed): decode-case all_reduce (tokens<world) via all_gather+sum

When tokens < world_size (decode, tokens==1) dim-0 chunking is impossible;
every rank all_gathers all world whole [tokens,hidden] contributions and
sums on-device. Unconditionally correct; layout-safe (whole-tensor
transfers only). HW-verified [1,4096] at 4 ranks reduces fully."
```

---

## Task 4: Correctness regression test + runner

**Files:**
- Create: `tests/distributed/test_allreduce_2d.py`
- Create: `run_allreduce_2d_verify.sh` (workspace root)

**Interfaces:**
- Consumes: the fixed `allreduce` (Tasks 2-3).

- [ ] **Step 1: Write the regression test (plain torchrun script form — NOT pytest, to avoid the vLLM engine build)**

`test_allreduce_2d.py` runs as a plain torchrun python script (per the harness gotcha: pytest+conftest+spyre_inference plugin builds a Qwen3 engine and times out). It reopens fd 1/2 to a per-rank log, `_lazy_init`, `init_process_group("cpu:gloo,spyre:spyreccl")`, then for each shape asserts async==sync equivalence AND full reduction:
```python
SHAPES = [(8, 4096), (6, 4096), (1, 4096), (32, 4096)]  # even, uneven, decode, larger
def run(shape):
    r, n = dist.get_rank(), dist.get_world_size()
    t = torch.ones(shape, dtype=torch.float16, device=DEVICE)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    c = t.to("cpu")
    want = float(n)
    full = bool((c == want).all().item())
    print(f"AR2D shape={shape} rank={r}/{n} FULL={'yes' if full else 'NO'} "
          f"first4={c.flatten()[:4].tolist()}", flush=True)
```
For TP subgroups: also run on a `dist.new_group(ranks=[0,1])` subgroup with the partial-abort/consensus guard pattern from `test_async_dispatch.py` so a failure never hangs.

- [ ] **Step 2: Write `run_allreduce_2d_verify.sh`** — mirror `diag_allreduce_split_coverage.sh` v3 (plain torchrun, per-rank logs, `set +u` around `env.sh`, vfio guard, `SPYRECOMMS_LOGLEVEL=error`), post-processing `AR2D` lines into a PASS/FAIL verdict (`FULL=yes` on all ranks for all shapes = PASS).

- [ ] **Step 3: Sync + ask the user to run at TP=2, TP=4, and TP=8 (HW checkpoints)**

```bash
cd /home/tmhoangt/spyre-multi
bash run_allreduce_2d_verify.sh 2
bash run_allreduce_2d_verify.sh 4
bash run_allreduce_2d_verify.sh 8   # if >=8 AIUs available; else note skipped
```
Expected: all shapes `FULL=yes` at every TP size. TP=2 confirms no regression to the previously-passing case; TP=4/8 confirm the fix.

- [ ] **Step 4: Commit (after user confirms all green)**

```bash
git add tests/distributed/test_allreduce_2d.py ../run_allreduce_2d_verify.sh
git commit -s --no-verify -m "test(distributed): 2-D all_reduce correctness regression (TP 2/4/8, uneven, decode, subgroup)"
```

---

## Task 5: Benchmark gate (data artifact)

**Files:**
- Create: `bench_allreduce_2d.sh` (workspace root)

**Interfaces:**
- Consumes: the fixed `allreduce`.

- [ ] **Step 1: Write `bench_allreduce_2d.sh`** — plain-torchrun script that times `dist.all_reduce` on `[num_tokens, 4096]` at TP=4 for representative prefill (`num_tokens` in {128, 512, 2048}) and decode (`num_tokens=1`), N iterations after warmup, printing p50/p95 per shape. Use `torch.spyre` synchronize (or the backend's wait) around each iteration so timing is real. Compare against a recorded baseline number for the (broken-but-fast) ring if available; otherwise just record absolute p50/p95 as the artifact.

```python
# per shape: warmup 5, time 50 iters of dist.all_reduce(ones([toks,4096]));
# print BENCH shape=[toks,4096] tp=4 p50_us=.. p95_us=..
```

- [ ] **Step 2: Sync + ask the user to run the benchmark (HW checkpoint)**

```bash
cd /home/tmhoangt/spyre-multi && bash bench_allreduce_2d.sh 4
```
Expected: prints p50/p95 for each shape. This is a DATA ARTIFACT (not pass/fail) — it sizes the perf gap for the tracked pipelined-ring follow-up. Record the numbers in the commit message + a note in the spec's follow-up section.

- [ ] **Step 3: Commit**

```bash
git add ../bench_allreduce_2d.sh
git commit -s --no-verify -m "bench(distributed): 2-D all_reduce latency gate (prefill+decode, TP=4)

Records p50/p95 for the layout-safe compose path to size the pipelined-ring
perf follow-up. Numbers: <paste from HW run>."
```

---

## Self-Review

**Spec coverage:**
- Unit 1 (≥2-D dispatch) → Task 2 Step 1. ✓
- Unit 2 (dim-0 chunking, torch-layer narrow+clone) → Task 2 Step 3. ✓
- Unit 3 (un-park reduce_scatter/exchange_uneven/subgroup) → Task 1. ✓
- Decode tokens<N (all_gather+local-sum default) → Task 3. ✓
- Testing #1 correctness regression → Task 4. ✓
- Testing #3 subgroup → Task 4 Step 1 (subgroup + consensus guard). ✓
- Testing #4 benchmark gate → Task 5. ✓
- Testing #5 no-regression (async/allgatherv/world-path g0_) → Task 1 Step 3. ✓
- Risk: subgroup g0_ comm_id HW regression → Task 1 Step 3 (explicit STOP gate). ✓

**Placeholder scan:** No TBD/TODO. Decode branch, chunking, compose all have real code. Benchmark numbers are a runtime artifact (correctly deferred to the HW run, not a code placeholder).

**Type consistency:** `allreduce_2d_compose(at::Tensor&, const AllreduceOptions&)` declared (Task 2 Step 2) and defined (Task 2 Step 3), extended (Task 3 Step 1) — consistent. `reduce_scatter`/`allgather` list-form signatures match Task 1 Interfaces and the WIP bodies verified in exploration. `ReduceScatterOptions`/`AllgatherOptions`/`ReduceOp::SUM` match the WIP's usage.

**Known open item for the implementer to confirm at Task 2:** whether `allgather`'s list form requires all output slots equal size (uniform) or supports the per-chunk sizes used in the uneven `tokens % world != 0` case — the WIP has a variable path via `exchange_uneven`, but if uneven all_gather slot sizing needs the variable branch, the compose must supply correctly-sized `ag_slots` (already done in Task 2 Step 3). If the variable allgather path rejects mismatched-total reassembly, fall back to uniform chunking (pad to `ceil(tokens/world)*world`) — flagged as the one implementation risk to verify on the first HW run.
