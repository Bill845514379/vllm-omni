# RFC: Async D2H + Stage Pipeline Async Transfer for Qwen3-Omni

## 1. Motivation

PR #3164 solved the semantic optimization of "which requests need D2H" on the Thinker side. But it did not address two types of **synchronous blocking**:

1. **Intra-step synchronous D2H**: When the batch still contains multimodal requests,
   `hidden_states.detach().to("cpu").contiguous()` blocks on the default stream,
   stalling the GPU at the end of every thinker decode step, negating the
   schedule/execute overlap that `AsyncScheduler` already enables.
2. **Inter-stage synchronous pipeline**: With async chunk streaming enabled, each chunk
   must traverse the full chain of
   `Thinker D2H → pickle → SHM write → SHM read → unpickle → CPU copy → sync H2D`
   — a complete set of synchronous operations that determine first-audio latency.

See README §Architecture at a Glance and §Background for full quantitative analysis.
This RFC provides production-grade APIs, change points, implementation phases, and
acceptance criteria.

### 1.1 Baseline contract

This RFC **assumes PR #3164 or equivalent changes have already landed**:

- `omni_final_stage_id` is propagated from frontend to scheduler / worker;
- Text-only requests no longer trigger downstream stage payloads;
- In mixed batches, only requests needing downstream payloads participate in
  `pooler_output` construction;
- `_process_additional_information_updates` preserves `req_ids_filter` semantics,
  not generating Talker / Code2Wav inputs for text-only requests.

Therefore, this RFC does not redesign "which requests need copying." If the
implementation branch does not yet include #3164, it must be rebased / cherry-picked
before starting P0; otherwise, D2H that should be skipped would be incorrectly
async-ified, and pure-text no-op validation would also fail.

## 2. Goals & Non-Goals

### Goals
- **G1**: Convert Thinker critical-path D2H to an async pipeline (`S_d2h` + pinned
  ring pool + `BatchPayloadFuture`), ensuring reusable pinned slots are not reused
  before async chunk background threads finish sending.
- **G2**: Unified `pooler_output` element data type — regardless of mixed batch or
  all-multimodal, downstream always sees `dict[str, BatchPayloadSliceRef | <plain>]`.
- **G3**: Async-ify the async chunk cross-process pipeline — pinned recv buffer,
  remove redundant `.cpu()` in stage processors, `AsyncH2DController` to overlap
  H2D with talker forward, optional SHM zero-copy replacing pickle.
- **G4**: Compatibility with PR #3164's `omni_final_stage_id` skip logic,
  `AsyncScheduler`, CUDA Graph, `omni_prefix_cache`, and existing
  `OmniChunkTransferAdapter` contracts.
- **G5**: Pure-text traffic and deployments without async chunk enabled are 100% no-op.
- **G6**: All async optimizations must have clear ownership boundaries: future release
  must only happen after the last consumer finishes materialize / serialize /
  `connector.put`.

### Non-Goals
- **NG1**: Do not change Talker / Code2Wav algorithms or RVQ tensor layouts.
- **NG2**: Do not rewrite msgspec for the overall `ModelRunnerOutput` serialization
  path. We only hook `_resolve_futures`.
- **NG3**: Do not require changes to vLLM upstream `GPUModelRunner` public API.
- **NG4**: Do not enable `S_d2h` / `S_h2d` on NPU platforms; NPU falls back to
  synchronous path.
- **NG5**: Do not depend on `OmniChunkTransferAdapter` and worker being in the same
  process. The recv-loop side can do opportunistic H2D enqueue, but the worker entry
  must retain a fallback materialize.

## 3. Background: What Each Step Actually Does

### 3.1 Intra-step D2H (within thinker worker)

`vllm_omni/worker/gpu_ar_model_runner.py::execute_model`,
`hidden_states.detach().to("cpu").contiguous()` blocks on the default stream.
PR #3164 separates the "all-text" and "sliced" fast/slow paths, but **they remain
synchronous**.

### 3.2 Inter-stage chunk pipeline (async chunk streaming mode)

Files involved:
- `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`
  - `OmniChunkTransferAdapter._send_single_request` calls
    `custom_process_next_stage_input_func` (i.e. `thinker2talker_async_chunk`) in
    the save_loop background thread, then calls `connector.put()`.
  - `OmniChunkTransferAdapter._poll_single_request` calls `connector.get()` in
    the recv_loop background thread, writing payload to
    `request.additional_information`.
- `vllm_omni/distributed/omni_connectors/connectors/shm_connector.py`
  - `SharedMemoryConnector.put` → `serialize_obj(data)` (pickle) → `shm_write_bytes`
  - `SharedMemoryConnector.get` → `shm_read_bytes` → `deserialize_obj` (unpickle)
- `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`
  - `thinker2talker_async_chunk` / `talker2code2wav_async_chunk` contain numerous
    `.detach().cpu()` calls.

Actual data path per chunk:

```text
┌─────────────────────── Thinker engine_core ─────────────────────┐
│ worker.execute_model returns pooler_output (Tensor or Future)   │
│   → ChunkAdapter.save_async(pooling_output=..., request=...)    │
│   → save_loop thread:                                           │
│       _send_single_request(task):                               │
│         payload_data = thinker2talker_async_chunk(...)          │
│           - .detach().cpu() (after P3 becomes CPU, no-op)       │
│           - construct OmniPayload                               │
│         connector.put(payload_data)                             │
│           - serialize_obj(payload_data) ← pickle.dumps          │
│             one full tensor → bytes copy                        │
│           - shm_write_bytes → /dev/shm/<key> (memcpy)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                       POSIX SHM segment
                              │
┌─────────────────────── Talker engine_core ──────────────────────┐
│ recv_loop thread:                                               │
│   _poll_single_request(req):                                    │
│     connector.get(key)                                          │
│       - shm_read_bytes (memcpy into host bytes)                 │
│       - deserialize_obj(bytes) ← pickle.loads, allocate new Tensor│
│     req.additional_information = payload                        │
│   → main loop schedules, adds to input_batch                    │
│   → talker worker.execute_model:                                │
│       extract tensor from additional_information                │
│       tensor.to(device, dtype) ← synchronous H2D                │
│       talker forward                                            │
└─────────────────────────────────────────────────────────────────┘
```

Per chunk: **4 redundant memcpy + 1 synchronous H2D + pickle call stack overhead**.
The last two stages (Talker → Code2Wav) repeat the same pattern.

## 4. Proposed Design

### 4.1 Overall dataflow (end-to-end)

> This section presents the complete end-to-end picture after P0–P5 are all landed,
> connecting every component from §4.2–§4.5 into a closed loop. Readers can treat
> this diagram as the single source of truth for the entire RFC.

#### 4.1.1 Participants and new type ownership

| Actor | Process | New types introduced | Lifetime owner |
|---|---|---|---|
| Frontend (`serving_chat`) | API server | — | — |
| Engine core (per stage) | One process per stage | — | — |
| Worker (per stage) | engine core subprocess | `AsyncD2HController` `AsyncH2DController` `PinnedRingPool` | worker instance |
| `OmniChunkTransferAdapter` | inside engine core, with save_loop / recv_loop background threads | `tensor_blob` writer/reader | adapter instance |
| Stage processor (`thinker2talker_async_chunk` etc.) | called inside save_loop thread | — | — |
| Pooler output top-level objects | worker → adapter | `BatchPayloadFuture` `BatchPayloadSliceRef` | single step lifetime |
| Cross-process carrier | POSIX SHM segment (`/dev/shm/<key>`) | `TensorDescriptor` (msgspec header) + raw blob | unlinked after both sender and receiver release |
| Receiver-side handle | recv_loop → worker | `StagePayloadHandle` `AsyncH2DFuture` | until chunk is consumed by talker forward |

#### 4.1.2 End-to-end dataflow (using an audio chunk as example)

```text
            ┌──────────┐                                       ┌──────────────┐
Client ────►│ HTTP/SSE │ (1)                              (17) │ audio stream │────► Client
            └────┬─────┘                                       └──────┬───────┘
                 │                                                    ▲
                 ▼                                                    │
        ┌──────────────────┐                                          │
        │ serving_chat     │ (2) modalities → final_stage_id           │
        │ async_omni_engine│     additional_info["omni_final_stage_id"]│
        └────────┬─────────┘                                          │
                 │                                                    │
═════════════════╪══════════════════ Stage 0 = Thinker ═══════════════╪═════════
                 ▼                                                    │
        ┌──────────────────┐                                          │
        │ Engine core      │ (3) schedule, dispatch                   │
        │ + AsyncScheduler │                                          │
        └────────┬─────────┘                                          │
                 ▼                                                    │
        ┌─────── Worker ────────────────────────────────────┐         │
        │ S_c (compute)            S_d2h (copy)             │         │
        │  ┌────────────┐                                   │         │
        │  │forward N   │(4)                                │         │
        │  │hidden (GPU)│                                   │         │
        │  └────┬───────┘                                   │         │
        │       │ event e_N                                 │         │
        │       └────────►(5) wait_event(e_N)               │         │
        │                  pool.acquire()                   │         │
        │                  pinned[N].copy_(src,             │         │
        │                    non_blocking=True)             │         │
        │                  event done_N                     │         │
        │                                                   │         │
        │  ┌─────────────────────────────────────────────┐  │         │
        │  │ pooler_output[i] = {                        │  │         │
        │  │   "hidden": BatchPayloadSliceRef(           │  │         │
        │  │       batch_future=BPF_N, req_id=rid),      │  │         │
        │  │   "embed.tts_bos": ... (same)               │  │         │
        │  │ }                                           │  │         │
        │  └─────────┬───────────────────────────────────┘  │         │
        │            │                                      │         │
        │            ▼  msgspec encode hook (6)             │         │
        │   _resolve_futures(seen):                         │         │
        │     for SliceRef → batch_future.slice_for(rid)    │         │
        │       (host_wait done_N once, lazy)               │         │
        │   release determined by output / adapter ownership│         │
        └────────┬──────────────────────────────────────────┘         │
                 │ pooler_output now: dict[str, host Tensor | <plain>]│
                 ▼                                                    │
       ┌──────── OmniChunkTransferAdapter ──────────────────┐         │
       │ save_loop thread:                                  │         │
       │   thinker2talker_async_chunk(...) (7)              │         │
       │     no .cpu() — host tensors pass through          │         │
       │   tensor_blob.write(payload):                      │         │
       │     - msgspec header → SHM "<key>_hdr"             │         │
       │     - per-tensor blob → SHM "<key>_blob_*"         │         │
       │     - cudaHostRegister segment as pinned           │         │
       │   adapter.put_req_chunk[req_id] += 1               │         │
       └────────────────────┬───────────────────────────────┘         │
                            │                                         │
═════════════════════ SHM transit (8) ═══════════════════════════════│
                            │  /dev/shm/<reqid>_<stage>_<chunk>       │
═════════════════════════════════════════════════════════════════════│
                            │                                         │
═════════════════════ Stage 1 = Talker ══════════════════════════════│
                            │                                         │
       ┌─────── OmniChunkTransferAdapter ───────────────────┐         │
       │ recv_loop thread:                                  │         │
       │   connector.get_handle(<key>) (9)                  │         │
       │     tensor_blob.read:                              │         │
       │       open SHM segments                            │         │
       │       cudaHostRegister                             │         │
       │       wrap as host Tensor (zero-copy view)         │         │
       │     → StagePayloadHandle(payload_dict, shm_blobs)  │         │
       │   handle.schedule_h2d_inplace(                     │         │
       │     ctrl=AsyncH2DController, device=cuda:0) (10)   │         │
       │   request.additional_information = handle          │         │
       └────────────────────┬───────────────────────────────┘         │
                            │                                         │
                            ▼                                         │
        ┌──────────────────┐                                          │
        │ Engine core      │ (11) chunk ready → schedule              │
        └────────┬─────────┘                                          │
                 ▼                                                    │
        ┌─────── Worker ────────────────────────────────────┐         │
        │ S_h2d (issued in step 10, overlaps with talker    │         │
        │        schedule)                                  │         │
        │   ┌──────────────┐                                │         │
        │   │async H2D N   │                                │         │
        │   │event done_H_N│                                │         │
        │   └──────────────┘                                │         │
        │                                                   │         │
        │ S_c                                               │         │
        │   ┌──────────────────────┐                        │         │
        │   │execute_model entry   │ (12)                   │         │
        │   │  materialize_gpu():  │                        │         │
        │   │    consumer_stream.  │                        │         │
        │   │    wait_event(       │                        │         │
        │   │      done_H_N)       │                        │         │
        │   │  talker forward N    │                        │         │
        │   └──────┬───────────────┘                        │         │
        │          │ output (GPU)                           │         │
        │          ▼                                        │         │
        │ S_d2h: same pattern as stage 0 (13)               │         │
        │   schedule_batch_copy → BatchPayloadFuture        │         │
        │   pooler_output → SliceRef → encode materialize   │         │
        │                                                   │         │
        │ end of step:                                      │         │
        │   handle.cleanup_host_blobs() (14)                │         │
        │   → SHM segment unlink                            │         │
        └──────────┬────────────────────────────────────────┘         │
                   │                                                  │
       ┌──────── OmniChunkTransferAdapter ──────────────────┐         │
       │ save_loop:                                         │         │
       │   talker2code2wav_async_chunk(...) (15)            │         │
       │   tensor_blob.write → SHM                          │         │
       └────────────────────┬───────────────────────────────┘         │
                            │                                         │
═════════════════════ SHM transit ═══════════════════════════════════│
                            │                                         │
═════════════════════ Stage 2 = Code2Wav ════════════════════════════│
                            │                                         │
       Same (9)→(14) pattern as Stage 1:                              │
         recv_loop: tensor_blob.read → StagePayloadHandle             │
                    schedule_h2d_inplace                              │
         worker:    materialize_gpu → forward → audio waveform (16)   │
                    cleanup_host_blobs                                │
                            │                                         │
                            ▼                                         │
                   audio bytes ────► via engine output channel ───────┘ (17)
```

#### 4.1.3 Step annotations (matching bracketed numbers in the diagram)

| # | Location | Key action | Related section |
|---|---|---|---|
| 1 | Client | OpenAI HTTP, `extra_body={"modalities":["audio"]}` | — |
| 2 | Frontend | `serving_chat` parses modalities, `async_omni_engine._apply_omni_final_stage_metadata` writes `omni_final_stage_id` | PR #3164 |
| 3 | Stage 0 engine | `OmniARScheduler.schedule_with_async` dispatches to worker | §4.4 |
| 4 | Stage 0 / S_c | thinker forward N, hidden_states on GPU | — |
| 5 | Stage 0 / S_d2h | `AsyncD2HController.schedule_batch_copy(hidden[:N_valid], ranges, needed)` | §4.2, §4.5 |
| 6 | Stage 0 host | msgspec encode / save_loop via hook → `_resolve_futures(seen)`; release determined by future ownership | §4.4, §4.5.4 |
| 7 | Stage 0 save_loop | `thinker2talker_async_chunk` no longer calls `.cpu()`, host tensor passthrough | §4.5.6 |
| 8 | SHM | `tensor_blob.write` writes header + blobs, `cudaHostRegister` registers as pinned | §4.5.4 |
| 9 | Stage 1 recv_loop | `connector.get_handle` → `StagePayloadHandle` | §4.5.5 |
| 10 | Stage 1 recv_loop | `handle.schedule_h2d_inplace` → immediately enqueues copy on `S_h2d` | §4.5.7 |
| 11 | Stage 1 engine | chunk ready, schedule talker step | §4.5.5 |
| 12 | Stage 1 / S_c | `materialize_gpu` makes forward stream `wait_event(done_H_N)`; no host wait | §4.5.7 |
| 13 | Stage 1 / S_d2h | Same form as (5), reusing the same pair of controllers | §4.5.7 |
| 14 | Stage 1 worker | end of step `cleanup_host_blobs` → SHM segments unlinked | §4.5.4, §4.5.8 |
| 15 | Stage 1 save_loop | `talker2code2wav_async_chunk` + `tensor_blob.write` | §4.5.6 |
| 16 | Stage 2 worker | code2wav forward produces audio waveform (most models run vocoder on GPU, final D2H for main process) | §4.5.7 |
| 17 | Frontend | audio bytes flow back to client via engine output | — |

#### 4.1.4 Three parallelism axes

Re-examining the diagram through the lens of "what overlap does concurrency provide,"
this RFC creates three groups of overlap:

**Axis A — Intra-step: thinker forward N+1 ⫼ thinker D2H N**
```
S_c    ─[fwd N]──────┬─[fwd N+1]──────┬─[fwd N+2]
                     │                │
S_d2h          [D2H N─────►]  [D2H N+1───►]
host                          [encode N]      [encode N+1]
```

**Axis B — Cross-process: thinker save_loop ⫼ talker schedule ⫼ talker H2D**
```
T0 save_loop  ─[encode N]─[wire write]─[SHM put]─►        [encode N+1]─...
T1 recv_loop                            ─[wire read]─[H2D enqueue]─►
T1 S_h2d                                              [async H2D N─►]
T1 schedule                                           [enqueue talker step]
T1 S_c                                                          [talker fwd N─►]
```

**Axis C — Talker internal: talker D2H ⫼ talker forward N+1 (same form as Axis A)**

The three axes maintain correct ordering through `cudaEvent` / `wait_event`; the host
only performs "lazy sync" at the msgspec encode line.

#### 4.1.5 Quick comparison with today (PR #3164 + AsyncScheduler)

| Item | Today | This RFC (P0–P5 all landed) |
|---|---|---|
| Thinker D2H | sync `to("cpu")` | async `S_d2h` + `BatchPayloadFuture` |
| `pooler_output` data type | mixed: dict-of-Tensor / batch-tensor dual form | always `dict[str, BatchPayloadSliceRef \| <plain>]` |
| Sender stage processor `.cpu()` | multiple times per chunk (even if no-op after P3) | completely removed |
| Wire protocol | `pickle.dumps` entire payload | msgspec header + SHM blob (zero-copy) |
| Receiver-side tensor source | `pickle.loads` allocates new host tensor | SHM segment direct wrap, 0 copy |
| Talker H2D | sync `tensor.to(device)` at forward entry | enqueued on `S_h2d` when chunk is queued, forward `wait_event` |
| SHM segment management | write once, read once, unlink | explicit ref-count, multi-stage chain won't prematurely unlink |
| Kill switch | none | `OMNI_ASYNC_D2H` / `OMNI_ASYNC_CHUNK` / `OMNI_ASYNC_CHUNK_WIRE` |

### 4.2 Key API (intra-step)

```python
# vllm_omni/worker/async_d2h.py  (new)

@dataclass
class BatchPayloadFuture:
    pinned: torch.Tensor                         # full-batch pinned host buffer
    event: torch.cuda.Event | None               # None = already complete
    pool: "PinnedRingPool"
    req_token_ranges: dict[str, tuple[int, int]]
    needed_req_ids: set[str]
    _refcount: int = 1                            # initial owner = ModelRunnerOutput
    def slice_for(self, req_id: str) -> torch.Tensor | None: ...
    def retain(self) -> "BatchPayloadFuture": ...
    def release(self) -> None: ...                # returns pool slot when refcount==0

@dataclass(slots=True)
class BatchPayloadSliceRef:
    batch_future: BatchPayloadFuture
    req_id: str

class PinnedRingPool:
    """Fixed-size pinned host buffer pool. Each slot size determined by
       max_num_tokens × hidden_dim."""
    def __init__(self, slot_shape: torch.Size, dtype: torch.dtype,
                 num_slots: int = 4): ...
    def acquire(self) -> torch.Tensor: ...     # take an unoccupied slot
    def release(self, tensor: torch.Tensor) -> None: ...

class AsyncD2HController:
    def __init__(self, vllm_config, copy_stream_priority: int = -1): ...
    def schedule_batch_copy(
        self,
        src: torch.Tensor,                      # length = num_valid_tokens
        req_token_ranges: dict[str, tuple[int, int]],
        needed_req_ids: set[str],
    ) -> BatchPayloadFuture: ...
```

### 4.3 Change points (intra-step)

| File | Change |
|---|---|
| `vllm_omni/worker/async_d2h.py` | New file, types and controller implementation |
| `vllm_omni/worker/gpu_ar_model_runner.py` | Replace `.to("cpu")` with `controller.schedule_batch_copy(...)`; `pooler_output[i]` is always a `dict` holding `BatchPayloadSliceRef` |
| `vllm_omni/worker/gpu_model_runner.py` | `_process_additional_information_updates` internal `.cpu()` calls also go through controller, one `BatchPayloadFuture` per multimodal sub-field; preserve `req_ids_filter` |
| `vllm_omni/v1/outputs/omni_model_runner_output.py` | Add msgspec encode hook: `_resolve_futures(seen)`; non-async-chunk path releases after encode, local async-chunk path retained/released by adapter |
| `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` | `save_async` retains futures before enqueue; `_send_single_request` calls `finally release` after `connector.put()` succeeds/fails |
| Config | `OMNI_ASYNC_D2H=0/1`, `OMNI_ASYNC_D2H_POOL_SLOTS`; YAML `omni_config.async_d2h` |

### 4.4 Unified pooler output data type

See README §"Pooler output unified data type." Key points:
- Single dict-of-mixed form, downstream no longer branches on type;
- Prefix-cache already-CPU path wrapped as `BatchPayloadFuture(event=None)`;
- Multimodal sub-fields each hold one `BatchPayloadFuture`;
- Future materialization and return are decoupled: encode / stage processor can
  materialize, pinned slot only returns to pool after the last consumer releases.

#### 4.4.1 Future ownership rule

The pinned tensor backing a `BatchPayloadFuture` is a reusable ring slot and cannot
be passed raw to background threads like a normal `tensor.to("cpu")` return value.
All paths must adhere to:

```text
worker creates future (ref=1)
  ├─ normal engine output encode:
  │    _resolve_futures(...) materialize
  │    finally future.release()
  └─ async chunk save_async:
       retain_futures(pooling_output)       # ref += 1 before enqueue
       worker/output owner release          # ref -= 1
       save_loop _send_single_request:
         _resolve_futures(...) materialize
         connector.put(payload_data)
         finally release_retained_futures   # ref -= 1, slot can be reused
```

Acceptance requirement: any payload passed to `OmniChunkTransferAdapter._pending_save_reqs`
must not contain bare pinned views that could be overwritten by the next step, unless
the adapter has already retained the corresponding future. Old paths that cannot
propagate the lease must clone into an independent CPU tensor before handing off to
background threads.

---

### 4.5 Stage Pipeline Async Transfer

This is the core of this RFC. It refactors the Thinker → Talker → Code2Wav chain's
"pickle + sync H2D + redundant `.cpu()`" using a unified abstraction. Scope spans 4
files: sender / wire / receiver / consumer worker.

#### 4.5.1 Current state re-examined

`OmniChunkTransferAdapter` already structures the transfer into two **background
threads** + queues:

| Thread | Role | Key calls |
|---|---|---|
| `save_loop` | sender | `_send_single_request → custom_func → connector.put` |
| `recv_loop` | receiver | `_poll_single_request → connector.get → req.additional_information=` |
| Main loop | engine scheduling | `worker.execute_model` reads `additional_information` |

This means:
- **Sender-side pickle/SHM write is not on the thinker forward critical path**,
  concurrent with the main loop;
- But **receiver unpickle / receive host buffer allocation, and talker worker H2D**
  are still in front of the talker forward critical path, blocking every chunk.

Therefore this RFC's optimization focus is on the **wire protocol** +
**receiver/talker side**; the sender side only gets minor changes (remove redundant
`.cpu()` and protocol adaptation).

#### 4.5.2 Invariants

I1. All tensor fields of a chunk must be on GPU and data-complete before being read
    by talker forward.

I2. `OmniChunkTransferAdapter.put_req_chunk` / `get_req_chunk` counter increment
    order unchanged; `meta.finished` termination semantics unchanged; `cleanup`
    conditions unchanged.

I3. D2H pinned ring slot lifecycle: write → all consumers materialize → last
    consumer release → return to pool. In async chunk mode, the save_loop background
    thread is an independent consumer; the slot must not be reused before
    `_send_single_request` finishes calling `connector.put()`.

I4. SHM segment lifecycle: write → receiver finishes read → delete lock_file
    (existing logic). With zero-copy introduced, **segments must not be deleted
    while the receiver holds tensor handles** — requires explicit ref-count. See §4.5.4.

I5. Tensors in `request.additional_information` can be released after being consumed
    once by the worker (already H2D-copied). But the AR chunk accumulation path
    `torch.cat`s into new host tensors; only when SHM/tensor references contributing
    to the accumulated result are no longer used by the request payload can they be
    reclaimed.

I6. When async chunk is off (sync mode), `AsyncH2DController.schedule_h2d`
    degrades to synchronous `tensor.to(device)`, wire protocol also degrades to
    pickle, behavior fully consistent with today.

I7. H2D enqueue cannot assume recv-loop and worker share a CUDA context / stream
    owner. If the adapter thread can safely access the worker-side
    `AsyncH2DController`, it may enqueue early; otherwise `StagePayloadHandle` only
    stores the host payload, and the worker `execute_model` entry does fallback
    schedule + materialize.

#### 4.5.3 Producer (sender) side

Files: `stage_input_processors/qwen3_omni.py`
+ `chunk_transfer_adapter.py::_send_single_request`

##### Current sender redundancy

Inside `thinker2talker_async_chunk` there is code like:

```python
"prefill": thinker_emb.detach().cpu(),
"tts_bos": thinker_embed.get("tts_bos").detach().cpu()
            if isinstance(thinker_embed.get("tts_bos"), torch.Tensor) else None,
...
```

After P1a-P3 land, `thinker_emb` is already a **pinned host tensor** (from
`BatchPayloadFuture.slice_for(rid).materialize()`), `.detach().cpu()` is a no-op
identity — but dispatch + one small memory alloc per call is still overhead, and
per chunk × multiple fields × multiple stages adds up measurably (~30 µs scale,
5–8% of single-chunk time at BS=1 streaming).

##### Change P4-S1: `_already_host_tensor`

```python
def _already_host_tensor(t: Any) -> torch.Tensor | None:
    if isinstance(t, torch.Tensor) and t.device.type == "cpu":
        return t
    return None
```

Replace all `x.detach().cpu()` patterns with:

```python
out = _already_host_tensor(x)
if out is None:
    out = x.detach().cpu()         # old path, exceptional fallback
```

After P3 lands, this fallback is never triggered in new code; keeping it ensures
correctness when async-d2h is off (`OMNI_ASYNC_D2H=0`).

##### Change P4-S2: sender stops using pickle

The payload before `connector.put(payload_data)` no longer depends on pickle.
Switch to unified wire encoding (§4.5.4): sender converts payload into
`{header_msgspec_bytes, blob_shm_segments}` two parts, each written to SHM.

#### 4.5.4 Wire protocol: replacing `serialize_obj`

New module: `vllm_omni/distributed/omni_connectors/wire/tensor_blob.py`

##### Header

`payload_data` is a nested dict, leaves are `torch.Tensor` / `int` / `str` /
`list`. We use msgspec to encode "non-tensor parts + tensor descriptors" into a
header:

```python
class TensorDescriptor(msgspec.Struct, frozen=True):
    shape: tuple[int, ...]
    dtype: str                       # e.g. "torch.bfloat16"
    blob_key: str                    # SHM segment key, /dev/shm/<key>
    is_pinned: bool                  # whether segment is registered as pinned host memory
```

When walking `payload_data`, replace tensors with `TensorDescriptor(blob_key=...)`,
msgspec-encode the entire header and write to one SHM segment (small, a few hundred
bytes).

This path does not reuse the existing `shm_read_bytes` helper: that helper copies
SHM content to `bytes` and immediately unlinks, conflicting with zero-copy tensor
view lifetime requirements. P5-W must provide an independent `tensor_blob`
read/write/cleanup API and fail-fast via magic bytes when encountering old
pickle/OmniSerializer payloads.

##### Blob

Each tensor gets its own SHM segment:

```python
def write_tensor_blob(t: torch.Tensor) -> TensorDescriptor:
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    blob_key = f"omni_blob_{uuid.uuid4().hex}"
    shm = shm_pkg.SharedMemory(name=blob_key, create=True, size=nbytes)
    # mmap'd memory can be registered as cuda host memory (pinned)
    register_cuda_host_memory(shm.buf, nbytes)        # cudaHostRegister
    np_view = np.ndarray(t.shape, dtype=torch_to_numpy(t.dtype), buffer=shm.buf)
    np_view[...] = t.numpy()                          # one memcpy from pinned src, no further copies
    return TensorDescriptor(t.shape, str(t.dtype), blob_key, is_pinned=True)
```

Note: **this one memcpy on the send side is necessary** (pinned host → SHM mapped
pinned host); but **it eliminates the internal copy + bytes object alloc from
pickle.dumps**, and the receive side will have zero tensor data memcpy.

##### Receiver decode

```python
def read_tensor_blob(desc: TensorDescriptor) -> torch.Tensor:
    shm = shm_pkg.SharedMemory(name=desc.blob_key)
    np_view = np.ndarray(desc.shape, dtype=torch_to_numpy(desc.dtype), buffer=shm.buf)
    t = torch.from_numpy(np_view)                     # zero-copy view
    if desc.is_pinned:
        # Segment already pinned by sender; receiver also registers to make this
        # process's page table treat it as pinned (cudaHostRegister is process-local)
        register_cuda_host_memory(shm.buf, np_view.nbytes)
    t._omni_shm = shm                                 # hold handle, prevent premature segment reclamation
    return t
```

##### Segment lifecycle & ref-count

`OmniChunkTransferAdapter` already assigns the payload to
`request.additional_information` after `_poll_single_request` completes. New
cleanup hook: after the worker confirms the H2D copy is complete and forward
no longer needs the host reference, call:

```python
async_handle.cleanup_host_blobs()       # delete SHM segments, unregister host memory
```

On the `is_payload_finished` termination chunk path, add the same cleanup after the
existing `cleanup` call. Multi-stage (thinker→talker→code2wav): each stage reclaims
upstream SHM after its H2D completes; if the chunk participates in AR `torch.cat`
accumulation, reclamation is deferred until the accumulated payload is overwritten
or request cleanup.

##### Degradation

If `OMNI_ASYNC_CHUNK_WIRE=pickle` (default before P5 lands), keep the old
`serialize_obj` path, only `_already_host_tensor` changes take effect.

#### 4.5.5 Receiver side

Files: `chunk_transfer_adapter.py::_poll_single_request` +
`shm_connector.py::get`.

##### Current state

```python
result = self.connector.get(...)            # bytes + meta
payload_data, size = result                 # already unpickled ← bytes->Tensor alloc
request.additional_information = payload_data
```

##### P4-R1: directly produce `StagePayloadHandle`

```python
@dataclass
class StagePayloadHandle:
    """After cross-process arrival, the receiver-side homomorphic handle.

    Dual of BatchPayloadFuture:
      - BatchPayloadFuture: GPU→host (D2H) async, wait on host
      - StagePayloadHandle:  host→GPU (H2D) async, wait on stream
    """
    payload_dict: dict[str, Any]                          # nested dict, leaves may be host Tensor
    h2d_futures: dict[Any, "AsyncH2DFuture"]              # already-scheduled H2Ds
    shm_blobs: list[shared_memory.SharedMemory]           # held SHM segments, deferred release
    needed: bool                                          # PR #3164: set after omni_final_stage_id check

    def schedule_h2d_inplace(self, controller: "AsyncH2DController",
                             device: torch.device) -> None:
        """Called immediately when chunk is enqueued, before worker forward."""
        self.h2d_futures = controller.schedule_h2d_dict(self.payload_dict, device)

    def materialize_gpu(self) -> dict[str, Any]:
        """Called at worker forward entry; returned dict has all tensors on GPU."""
        return _walk_replace_with_h2d_results(self.payload_dict, self.h2d_futures)

    def cleanup_host_blobs(self) -> None:
        for shm in self.shm_blobs:
            try: shm.close(); shm.unlink()
            except FileNotFoundError: pass
```

`_poll_single_request` changes to:

```python
result = self.connector.get_handle(...)        # returns StagePayloadHandle (new interface)
if result is None: return False
handle, size = result
self.get_req_chunk[req_id] += 1
# If adapter can safely share controller with worker, schedule H2D early here;
# otherwise just store handle, worker execute_model entry does fallback schedule.
if self._h2d_controller is not None:
    handle.schedule_h2d_inplace(self._h2d_controller, self._device)
request.additional_information = handle
```

`get_handle` is a new method on `OmniConnectorBase`; old `get` is preserved for
compatibility with non-SHM connectors.

##### P4-R2: existing `_update_request_payload` `torch.cat` in AR mode

This is AR streaming accumulation of per-chunk hidden states. Current implementation:

```python
new_val[qual] = torch.cat([origin_sub[qual], value], dim=0)
```

After changes, `value` is already the host tensor from
`StagePayloadHandle.payload_dict[...]`. `torch.cat` uses the host CPU path, same
behavior as today; no change needed. But note: **accumulated tensors must not have
their segments unlinked**, so `_update_request_payload` increments the segment hold
count for tensors contributing to the accumulated result by 1, with corresponding
cleanup -1. Simple implementation: also store `value._omni_shm` in
`request_payload[req_id]['_shm_refs']` list, release uniformly on `cleanup_sender`.

#### 4.5.6 Stage processor side

`thinker2talker_async_chunk` now receives `pooling_output` (actually the sender
worker's output `pooler_output[i]`) already as host tensor. New changes:

```python
def thinker2talker_async_chunk(transfer_manager, pooling_output, request, is_finished):
    if not isinstance(pooling_output, dict):
        return None
    thinker_layers = ...
    thinker_emb = _layer_tensor(thinker_layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(thinker_layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None or thinker_hid is None:
        return None

    # ★ No more .detach().cpu() — after P3, thinker_emb is guaranteed host tensor
    if chunk_id == 0:
        payload = {
            "embed": {
                "prefill": thinker_emb,                                  # was .detach().cpu()
                "tts_bos": _maybe_host(thinker_embed.get("tts_bos")),
                "tts_eos": _maybe_host(thinker_embed.get("tts_eos")),
                "tts_pad": _maybe_host(thinker_embed.get("tts_pad")),
            },
            "hidden_states": {"output": thinker_hid},                    # was .detach().cpu()
            ...
        }
    ...
```

`_maybe_host(t)` = `t if (isinstance(t, torch.Tensor) and t.device.type=='cpu') else None`.
Keep None placeholder so the wire protocol header can still correctly express
"this field is absent."

Same changes for `talker2code2wav_async_chunk`.

#### 4.5.7 Consumer worker side: `AsyncH2DController`

File: `vllm_omni/worker/async_h2d.py` (new).

##### API

```python
@dataclass
class AsyncH2DFuture:
    gpu_tensor: torch.Tensor                # pre-allocated GPU buffer (caching alloc)
    event: torch.cuda.Event                 # recorded on S_h2d
    src_host: torch.Tensor                  # holds host tensor reference, prevents premature SHM unlink

    def wait_and_get(self, consumer_stream: torch.cuda.Stream | None = None) -> torch.Tensor:
        """Make consumer (forward) stream wait for this H2D to complete, then return GPU tensor.

        Key: uses stream wait_event, does NOT host-wait."""
        if consumer_stream is None:
            consumer_stream = torch.cuda.current_stream()
        consumer_stream.wait_event(self.event)
        # ⚠ src_host must not be modified after this, until forward is done;
        #    forward completion is guaranteed by consumer_stream ordering,
        #    worker releases src_host uniformly at end of step.
        return self.gpu_tensor

class AsyncH2DController:
    def __init__(self, device: torch.device, stream_priority: int = -1): ...

    def schedule_h2d(self, src_host: torch.Tensor, dst_dtype: torch.dtype | None = None
                     ) -> AsyncH2DFuture:
        """src_host must be pinned (or page-locked via cudaHostRegister)."""
        with torch.cuda.stream(self.h2d_stream):
            gpu = torch.empty_like(src_host, device=self.device,
                                   dtype=dst_dtype or src_host.dtype)
            gpu.copy_(src_host, non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.h2d_stream)
        return AsyncH2DFuture(gpu_tensor=gpu, event=ev, src_host=src_host)

    def schedule_h2d_dict(self, payload: dict[str, Any], device: torch.device
                          ) -> dict[Any, AsyncH2DFuture]:
        """Walk nested dict, schedule_h2d each host tensor leaf."""
        ...
```

##### Landing point

Talker / Code2Wav worker `execute_model` entry adds:

```python
def execute_model(self, scheduler_output, ...):
    # Fallback schedule for handles not yet scheduled (non-SHM path)
    self._schedule_pending_h2d_for_input_batch()

    # Before forward, realize H2D futures into GPU tensors (stream wait, no host block)
    self._materialize_h2d_for_input_batch()

    # ... original forward path, sees additional_information.<tensor fields> already on GPU
    return super().execute_model(scheduler_output, ...)
```

`_schedule_pending_h2d_for_input_batch` is the fallback; on the normal SHM path,
`_poll_single_request` has already called `schedule_h2d_inplace` in the recv_loop
thread.

##### Stream relationships

- `S_h2d` and `S_compute`: use `consumer_stream.wait_event(future.event)` to make
  forward stream wait for H2D completion; **no host wait**.
- `S_h2d` and `S_d2h`: this stage's worker-internal D2H and incoming next-stage H2D
  are two independent streams, not affecting each other.
- Release: `src_host` reference held until end of step, released uniformly by
  `_release_finished_h2d_inputs` (also the trigger point for SHM segment unlink).

#### 4.5.8 End-to-end timing & invariants

##### Steady-state pipeline (one chunk)

```text
   chunk N                  chunk N+1                   chunk N+2
T₀──────────T₁           T₂──────────T₃             T₄──────────T₅

Thinker S_c    [forward N─────►]      [forward N+1───►]      [forward N+2─►]
Thinker S_d2h        [D2H N───────►]       [D2H N+1───►]
Save loop                  [pkt+SHM─►]           [pkt+SHM─►]
                            (sender)             (sender)
SHM transit                       ↘                   ↘
Recv loop                          [SHM read+H2D shed]   [SHM read+H2D shed]
Talker S_h2d                            [H2D N─►]              [H2D N+1─►]
Talker S_c                                  [forward N───────►]    [forward N+1...
```

Key cross-constraints:
- I1: `Talker S_c.wait_event(H2D_N.event)` guarantees forward N sees complete tensor
- I3: `cleanup_host_blobs(N)` must happen after `Talker S_c` step N completes
  (GPU tensor consumed); implemented via worker's end-of-step
  `_release_finished_h2d_inputs`, attached after step output processing

##### Async chunk off (non-streaming)

`schedule_h2d` degrades to `gpu = src_host.to(device, non_blocking=False)`, event
is already complete; behavior fully consistent with sync path.

#### 4.5.9 Backpressure & failure

##### Backpressure

- Pinned recv buffer naturally rate-limits via ring pool + segment ref-count: pool
  full → `connector.get_handle` returns None, recv_loop waits for next poll.
- Talker engine's input_batch already has max_num_seqs cap; H2D future count is
  naturally bounded.

##### Failure handling

| Scenario | Handling |
|---|---|
| `cudaHostRegister` fails (kernel ulimit insufficient) | log warn, fall back to P4 sync H2D path, no correctness impact |
| sender wrote SHM but segment cleaned externally before read | `connector.get_handle` catches `FileNotFoundError`, returns None, recv_loop retries until timeout |
| receiver holds handle but worker process crashes | SHM segment via `shm.unlink` only reclaimed after all refs released; worker crash → monitor spawns new process, new process re-`cudaHostRegister`s |
| H2D fails (GPU OOM) | future.event never ready → `wait_and_get` throws `cuda.errors.CudaError` before forward, worker catches and fails that request |
| pickle path still active (`OMNI_ASYNC_CHUNK_WIRE=pickle`) during H2D | controller still used, fallback uses pickle-deserialized host tensor as src, correct but ZC benefit reduced |

##### Debug switches

- `OMNI_ASYNC_CHUNK=0` → disable entire §4.5, keep §4.1–4.4 intra-step optimizations.
- `OMNI_ASYNC_H2D_DEBUG=1` → `schedule_h2d` immediately host-waits event, degrades
  async to sync, for bisecting numerical differences.
- `OMNI_ASYNC_CHUNK_WIRE=pickle|tensor_blob` → force wire protocol.
- `record_function_or_nullcontext("async_h2d:schedule")` and `("async_h2d:wait")`
  let nsys / torch.profiler directly see H2D intervals.

#### 4.5.10 Compatibility with existing `OmniChunkTransferAdapter` contracts

| Existing contract | Compatibility |
|---|---|
| `put_req_chunk[req_id]` monotonically increasing | unchanged (still `+= 1` at end of `_send_single_request`) |
| `meta.finished` termination semantics | unchanged (carried as `meta` dict in wire header) |
| `_update_request_payload` AR `torch.cat` accumulation | unchanged; segment refs added to `_shm_refs` |
| `cleanup_sender` / `cleanup_receiver` timing | unchanged; new `cleanup_host_blobs` reclaims segments after H2D complete |
| `custom_process_next_stage_input_func` contract | unchanged; processor now produces host tensor passthrough, no longer triggers `.cpu()` |
| `connector.get` (old) | preserved; `get_handle` is a new method, only `SharedMemoryConnector` implements it, other connectors auto-degrade via old path |

---

## 5. Implementation Phases

### 5.1 Overview

P0–P5 are phase codenames — each phase maps to an independent, individually-PR-able
small change step, convenient for gradual rollout and rollback. Overall structure:
two segments by scope, further subdivided by code surface area:

- **P0–P3 = Intra-step**: only optimize thinker worker single-process GPU↔host copy,
  no cross-process pipeline changes; benefits any deployment form (including
  non-streaming).
- **P4–P5 = Inter-stage**: only effective with async chunk streaming enabled
  (Qwen3-Omni TTS path).
- Letter suffixes (P4-S/R, P4b, P5-W0/W/N) = independently committable sub-steps
  within a phase, each independent PR with independent perf data.
- Every phase has an ENV switch (`OMNI_ASYNC_D2H` / `OMNI_ASYNC_CHUNK` /
  `OMNI_ASYNC_CHUNK_WIRE`); flipping one variable rolls back to the previous phase
  without reverting code.

### 5.2 Phase table

| Phase | Short name | One-sentence goal | Main files changed | Effort | Marginal gain | Risk |
|---|---|---|---|---|---|---|
| **P0** | baseline guard + pinned copy PoC | Confirm #3164 landed; replace sync `.to("cpu")` with explicit pinned dst + `copy_(non_blocking=True)`, no cross-step buffer reuse | `gpu_ar_model_runner.py` | 1–2d | TPOT −3 ~ −5% | Low |
| **P1a** | Future ownership | Introduce `BatchPayloadFuture` / `SliceRef`, first solve retain/release and async chunk save_loop lifetime, prevent leaking bare views to background threads | `worker/async_d2h.py` (new), `gpu_ar_model_runner.py`, `chunk_transfer_adapter.py` | 2–3d | Correctness prerequisite | Medium |
| **P1b** | AsyncD2HController | Introduce dedicated `S_d2h` stream + lazy materialize; `record_stream` protects src, host only waits at materialize point | `worker/async_d2h.py`, `gpu_ar_model_runner.py`, `omni_model_runner_output.py` | 3–5d | TPOT −8 ~ −15% | Medium |
| **P2** | PinnedRingPool hardening | Fixed-size ring pool + capacity cap + ENV/CLI config + pool leak metrics + long-run stability | `worker/async_d2h.py` | 3–5d | High QPS stability | Medium |
| **P3** | mm sub-field unification | `_process_additional_information_updates` internal all `.cpu()` also through controller, one future per multimodal field | `gpu_model_runner.py` | 2–3d | TPOT another −5 ~ −10% | Low |
| **P4-S1/S2** | sender de-duplication | `thinker2talker_async_chunk` etc. remove redundant `.detach().cpu()`; optional wire protocol adaptation | `qwen3_omni.py` etc. stage processors | 1–2d | Streaming first-audio −3 ~ −5% | Low |
| **P4-R1** | receiver handle | `StagePayloadHandle` + `connector.get_handle`; recv-loop produces host tensor/handle, no hard dependency on early H2D | `chunk_transfer_adapter.py`, `connectors/base.py`, `shm_connector.py` | 3–5d | Streaming first-audio −5 ~ −10% | Medium |
| **P4b** | AsyncH2DController | Dual controller: dedicated `S_h2d` + future; worker entry fallback schedule, early enqueue from recv-loop when controller safely shareable | `worker/async_h2d.py` (new), consumer worker `execute_model` entry | 3–5d | Streaming first-audio −10 ~ −20% | Medium-High |
| **P5-W0** | tensor_blob PoC | Independently validate `SharedMemory` + `cudaHostRegister` + zero-copy tensor view + crash cleanup, not connected to production path | `wire/tensor_blob.py` (new), tests | 2–4d | De-risk P5 | High |
| **P5-W** | tensor_blob wire | Replace pickle: msgspec header + SHM segments directly back tensor data (`cudaHostRegister`), eliminating pickle memcpy | `wire/tensor_blob.py`, `shm_connector.py` | 1–2w | Streaming first-audio −5% | High |
| **P5-N** | NPU adaptation | NPU platform `S_d2h` / `S_h2d` rewritten with `torch_npu.npu.Stream`, else degrade to sync | `platforms/npu/worker/*.py` | 1w | Cross-platform consistency | Medium |

### 5.3 Dependency graph

```text
                       ┌── P0 ── P1a ── P1b ── P2 ── P3 ─┐  (Intra-step, any deployment)
                       │                          │
   PR #3164 (baseline)─┤                          │
                       │                          │
                       └── P4-S1/S2 ──────────────┤
                                                  │
                            P4-R1 ─── P4b ────────┤  (Inter-stage, async chunk)
                                                  │
                                      P5-W0 ── P5-W ─┤
                                                  │
                                            P5-N ─┘  (NPU cross-platform)
```

- **P0 must validate #3164 baseline**: without #3164's request gating, pure-text
  no-op and mixed-batch semantics are not valid.
- **P1a must precede P1b**: first prove pinned slot ownership won't be corrupted
  by async chunk background threads, then put the copy on an async stream.
- **P3 must precede P4-S**: sender `.cpu()` removal depends on P3 having made
  leaf tensors guaranteed host tensors.
- **P4-R1 must precede P4b**: `AsyncH2DController`'s src source is the host tensor
  exposed by `StagePayloadHandle`.
- **P4b and P5-W are independent**: P4b can be enabled even with the old pickle
  path, since `pickle.loads`'d host tensors can also be `cudaHostRegister`ed and
  fed to H2D; P5-W further saves pickle's redundant alloc.
- **P5-W0 must precede P5-W**: zero-copy SHM mmap / unlink / resource_tracker
  behavior must be hardened with a standalone PoC before connecting to the connector.
- **P5-N is decoupled from all phases above**: NPU platform just replaces
  `cuda.Stream` with `torch_npu.npu.Stream`, logic unchanged.

### 5.4 PR & acceptance criteria

Each phase must satisfy:

1. **Independent PR**, commit message prefixed `[perf][async-d2h][P<phase>] ...`.
2. **With perf data**: at least one set of nsys or `torch.profiler` screenshots +
   `tests/dfx/perf/` table comparing against previous phase numbers.
3. **Disableable**: via corresponding ENV, flip back to previous phase behavior,
   CI passes both modes.
4. **No regression**: pure-text baseline (`test_qwen3_omni_vllm_text`) any
   `mean_ttft_ms` deviation ≤ 2%.
5. **Lifetime proof**: any phase introducing reusable pinned / SHM buffers must
   have an async chunk delayed-consumption test verifying that post-enqueue next-step
   overwrite does not corrupt queued payload.
6. **Stability**: P2, P4-R1, P4b, P5-W must additionally run a 4h long-run (see §11.3).

## 6. Performance Expectations

Test setup: H100 / Qwen3-Omni-30B-A3B / `request_rate=1.0` / streaming TTS
(`extra_body={"modalities":["audio"]}`).

| Stage | mean TPOT | First-audio TTFA p50 | First-audio TTFA p99 |
|---|---|---|---|
| baseline (PR #3164) | 35 ms | 600 ms | 1100 ms |
| + P1a-P3 (intra-step async D2H) | 30 ms (-14%) | 580 ms (-3%) | 1050 ms (-5%) |
| + P4-S1/S2 (remove redundant cpu) | 30 ms | 565 ms (-6%) | 1010 ms (-8%) |
| + P4-R1 (pinned recv handle) | 30 ms | 540 ms (-10%) | 970 ms (-12%) |
| + P4b (AsyncH2DController) | 30 ms | 480 ms (-20%) | 870 ms (-21%) |
| + P5-W (tensor_blob zero-copy) | 30 ms | 450 ms (-25%) | 820 ms (-25%) |

Non-streaming (`modalities=["text"]`): all phases are no-op, unchanged.

Micro-bench: thinker single-step only, compare `nsys` whether `aten::copy_` sync
segment is gone (should not be visible after P1+).

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| #3164 not merged or implementation branch not rebased | P0 entry check: must prove text-only does not construct downstream payload, mixed batch only constructs payload for downstream reqs |
| reusable pinned slot overwritten while async chunk background thread still consuming | `BatchPayloadFuture.retain/release`; `save_async` retains before enqueue, `_send_single_request` finally releases; old paths that can't propagate lease must clone |
| Pinned memory exhaustion | `PinnedRingPool.num_slots` cap; fallback when `cudaHostRegister` fails for SHM segment |
| `record_stream` missed, src prematurely reclaimed by caching allocator | `AsyncD2HController.schedule_batch_copy` internally calls uniformly; fuzz test + `empty_cache` repeated triggers |
| msgspec encode ordering vs `_resolve_futures` double-walk performance | single walk complete + try/finally release; test asserts `time.perf_counter` < 5 µs / output |
| recv-loop thread cannot safely access worker CUDA stream/controller | `StagePayloadHandle` allows unscheduled state; worker `execute_model` entry does fallback schedule + materialize |
| AR chunk accumulation `torch.cat` invalidates early H2D future | first complete `_update_request_payload` accumulation, then schedule H2D on accumulated result; SHM refs consumed by cat deferred to request cleanup |
| SHM segment cleaned by external process (docker restart, `/dev/shm` full) | `connector.get_handle` catches `FileNotFoundError` → recv_loop retries + warn |
| `cudaHostRegister` segment re-registered across multiple stages | registration is process-local, each process independently registers; segment ref-count prevents premature unmap |
| P4b async H2D and talker forward ordering errors | `consumer_stream.wait_event(future.event)` enforces order, unit tests cover fast-write + slow-read fuzz |
| pickle path and tensor_blob path dual-run bugs | Wire protocol adds magic bytes, wrong protocol fail-fast immediately |
| NPU does not support `cudaHostRegister` | `platforms/npu/...` provides fallback; P5-N completed separately |

## 8. Alternatives Considered

1. **GPU-side IPC (cudaIpcOpenMemHandle)**: let talker directly read thinker GPU
   memory pointer. Only feasible on same GPU same process; tightly coupled to
   vllm-omni multi-stage multi-process architecture, high rollback cost.
2. **NCCL Send/Recv cross-stage GPU→GPU**: only effective when talker is on GPU,
   and depends on NCCL topology configuration; poor generality.
3. **Completely replace msgspec with Arrow Flight**: too heavy, incompatible with
   existing `OmniConnectorBase` abstraction.
4. **Keep sync + multi-worker instance parallelism**: low ROI, PP already covers this.

The async + tensor_blob approach is the minimum-invasiveness path best suited to
the existing architecture.

## 9. Migration & Rollout

- Config defaults:
  - `OMNI_ASYNC_D2H=0` (after P0-P3 land) → `=1` enabled by default (after two
    rounds of nightly).
  - `OMNI_ASYNC_CHUNK=0` → `=1` (after all P4 phases land).
  - `OMNI_ASYNC_CHUNK_WIRE=pickle` → `=tensor_blob` (after P5-W lands + one round
    of stability).
- CI: `tests/dfx/perf/tests/test_qwen_omni.json` add three mirrored config cases:
  - `OMNI_ASYNC_D2H=1` (streaming off)
  - `OMNI_ASYNC_CHUNK=1, OMNI_ASYNC_CHUNK_WIRE=pickle`
  - `OMNI_ASYNC_CHUNK=1, OMNI_ASYNC_CHUNK_WIRE=tensor_blob`
- Stability: `tests/dfx/stability/` add 4h long-run, monitor `/dev/shm` segment
  count, `pool.in_use_bytes`, host RSS, `cuda_mem_get_info()` trend.
- Rollback: for any phase with issues, single ENV flip disables it, no code revert
  needed.

## 10. Open Questions

1. Does `cudaHostRegister` on SHM segments conflict with
   `multiprocessing.shared_memory`'s implicit mmap? Needs PoC validation; if
   conflict, switch to `posix_ipc` for direct mmap control.
2. Does Python `multiprocessing.resource_tracker` auto-unlink segments while
   receiver still holds zero-copy tensor views? P5-W0 must cover process normal
   exit, crash, and parent process cleanup three paths.
3. Does the `tensor_blob` protocol need to support sparse tensors / nested tensors?
   Current Qwen3-Omni payload doesn't have them, skip for now, leave hook.
4. When P4b has multiple H2D futures in-flight simultaneously, does it need
   separate `S_h2d_high` / `S_h2d_low` to differentiate chunk sizes? Decide after
   micro-bench.
5. Code2Wav stage output is audio waveform (CPU tensor, returned to user), does it
   need a reverse D2H async pipeline? Current use case: Code2Wav directly runs
   vocoder on CPU or GPU→CPU volume is small, skip for now; leave for next version
   of §4.5.

## 11. Test Plan

### 11.1 Unit (`tests/test_async_d2h.py`, `tests/test_async_h2d.py`)
- `BatchPayloadFuture.slice_for` bit-equal vs. original `.to("cpu")`.
- `BatchPayloadFuture.retain/release`: run next step immediately after async chunk
  `save_async` enqueue to overwrite ring slot; background `_send_single_request`
  still reads old step data.
- Pool full `acquire` blocks + release wakes.
- `record_stream` prevents reclamation: allocate large batch of src tensors, force
  `empty_cache`, verify `materialize` data correct.
- `PayloadFuture.__del__` returns buffer on exception path.
- `AsyncH2DFuture.wait_and_get` data correct under fast-write+slow-read fuzz.
- `StagePayloadHandle` unscheduled: worker entry fallback schedules; already
  scheduled: no duplicate H2D.
- `tensor_blob` protocol: 1MB / 1GB tensor bidirectional round-trip bit-equal;
  segment truly unlinked after `cleanup_host_blobs`.

### 11.2 Integration (`tests/dfx/perf/`)
- `test_qwen3_omni_vllm_text` no regression (any mean_ttft deviation ≤ 2%).
- `test_qwen3_omni_streaming_audio` (new): 3 config groups (pickle / tensor_blob /
  pickle+sync) first-audio TTFA p50/p99 + audio quality (WER) no regression.
- `test_qwen3_omni_mixed_batch` (new): 50% text-only + 50% audio, verify
  `pooler_output[i]` data type consistency + first-audio time not dragged by
  text-only.
- `test_qwen3_omni_async_chunk_delayed_send` (new): mock save_loop delay
  `_send_single_request`, verify reusable pinned buffer does not corrupt queued
  payload.

### 11.3 Stability (`tests/dfx/stability/`)
- 4h long-run: monitor
  - `/dev/shm` segment count stable (no accumulation)
  - host RSS not exceeding baseline + 1GB
  - `cuda_mem_get_info` not declining
  - `pool.in_use_bytes` not exceeding `num_slots * slot_size`

---

_End of RFC_
