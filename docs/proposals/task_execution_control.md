# Task Execution Control

## Status

This document describes the proposed cooperative cancellation and progress
contract introduced by local commit `e568994`. The API is intentionally generic
to audio.cpp. Qwen3-ASR offline inference is the first adopter and is validation
evidence, not part of the framework-level contract.

## Motivation

`TaskRequest` previously carried model inputs and options but had no standard way
for an embedding application to:

- request cancellation after inference had started;
- observe coarse model stages or iterative decode progress;
- propagate cancellation into a backend graph which supports abort callbacks;
- distinguish cooperative cancellation from an ordinary provider failure.

Application-specific status codes, event loops, UI dispatch, and persistence do
not belong in this layer. The runtime needs only a synchronous per-request control
object which model sessions may observe.

## Framework API

`TaskRequest::execution_control` optionally references a shared immutable
`TaskExecutionControl`:

```cpp
struct TaskExecutionControl {
    std::function<bool()> cancellation_requested;
    std::function<void(const TaskProgress &)> progress;
};
```

`TaskProgress` contains an overall fraction, semantic stage name, and optional
stage-local completed/total units. A model integration owns its stage vocabulary
and is responsible for reporting monotonic fractions for one request.

The runtime provides:

- `task_cancellation_requested()` for non-throwing observation;
- `throw_if_task_cancelled()` for stage and loop boundaries;
- `report_task_progress()` for finite, clamped progress events;
- `TaskCancelled` as the C++ task-level cancellation signal;
- `ScopedTaskExecutionControl` for backend abort callback installation and
  deterministic cleanup.

No model is implicitly cancellable merely because the field exists. Each model
must add observation points and backend scope ownership appropriate to its run
mode, then add model-specific tests before advertising the capability.

## Callback Contract

Both callbacks execute synchronously on the inference thread.

- The control object and its `std::function` fields must remain immutable until
  the request finishes.
- A cancellation callback should return quickly and should not enter the same
  session. If it throws, the runtime fails closed and treats the request as
  cancelled.
- A progress callback may copy or dispatch the event elsewhere. If it throws,
  the exception propagates to the task caller as an application callback
  failure; it is not silently converted into cancellation.
- `TaskProgress::stage` is owned by the event supplied to the callback.
- `total == 0` means that a stage-local unit total is unavailable.

The runtime does not create a callback thread, queue events, or select an
application dispatcher. Those policies belong to the embedding application.

## Backend Contract

`ScopedTaskExecutionControl` queries the current backend registry for the
standard `ggml_backend_set_abort_callback` procedure. A backend which does not
export the procedure still receives model-level cancellation checks; it simply
cannot stop an in-flight graph.

The scope owns the raw callback-data lifetime through a shared reference and
clears the backend callback on every exit path. Scopes must not overlap or nest
on the same `ExecutionContext`, and callback installation/removal must not race
graph execution on that context. Higher-level sessions are expected to serialize
use of one execution context.

CPU uses ggml's existing abort callback implementation.

Metal exports the same registry procedure. While no callback is installed, its
existing asynchronous throughput path is unchanged. While a callback is
installed, graph submission is divided into command-buffer segments:

1. encode all segments;
2. submit and complete the main segment;
3. poll cancellation before each remaining segment;
4. commit the next segment only when cancellation is not requested;
5. release unsubmitted buffers on abort.

Already submitted GPU work completes. Cancellation latency is therefore bounded
by the current command-buffer segment, not guaranteed to be immediate. The
abortable path is deliberately synchronous at those boundaries. GPU capture is
stopped on success, abort, and newly handled command-buffer failure exits.

## First Model Integration

Qwen3-ASR offline inference installs one backend scope around `run()` and checks
cancellation at frontend, prompt, audio encoder, thinker prefill, per-token
decode, postprocess, and result boundaries. It reports overall stage fractions
and token completed/total units during decode. Multi-chunk requests map each
chunk's local progress into a non-overlapping global range.

Streaming Qwen3-ASR and other model families are not covered by this first
integration. They require separate scope/lifetime decisions and tests.

## SenseVoice Offline Integration

The local follow-up candidate applies the same contract to SenseVoice offline
inference without changing its buffered-streaming behavior. One backend scope
owns the complete offline `run()` call. The session checks cancellation before
work, between frontend/encoder/decode stages, and before publishing a result;
multi-chunk requests map each chunk into a non-overlapping fraction range.

The stage vocabulary is `sense_asr.starting`, `sense_asr.frontend`,
`sense_asr.encoder`, `sense_asr.decode`, and `sense_asr.complete`. A backend
graph failure is translated to `TaskCancelled` only while cancellation is
actually requested. Buffered streaming deliberately does not reuse this
per-offline-request progress contract: its start/push/finalize lifetime remains
a separate review item.

## Failure Mapping

Model integrations should preserve ordinary provider errors. If a backend graph
returns a generic failure while the cancellation callback is requested, the
session may translate that failure to `TaskCancelled`. Without a requested
cancellation, the original failure must propagate.

Cancellation is cooperative. A caller must not assume that observing a request
immediately stops worker threads or already submitted GPU commands.

## Reviewable Patch Split

The local prototype is one commit for integration testing. An upstream series
should be presented as three independently reviewable changes:

1. framework types, helpers, generic backend procedure lookup, and CPU tests;
2. Metal registry export, segmented abort path, capture/error cleanup, and Metal
   backend reuse test;
3. Qwen3-ASR offline observation points, progress mapping, and model regression
   evidence.
4. SenseVoice offline scope/observation points and model regression evidence.

This split keeps the generic runtime contract independent from one model and
makes the ggml Metal behavior change explicit to backend reviewers.

## Validation

The framework unit test verifies:

- progress lower/upper clamping and event fields;
- rejection of non-finite fractions;
- progress callback exception propagation;
- fail-closed cancellation callback exceptions;
- CPU and Metal abort callback discovery;
- cancellation after the first backend observation returns
  `GGML_STATUS_ABORTED`;
- clearing the scope allows the same graph/backend to run successfully again.

The Qwen3-ASR integration evidence additionally verifies deterministic output,
in-flight cancellation, progress monotonicity, repeated session reuse, repeated
model/session load-run-unload, and serialized same-session calls in the embedding
framework.

The SenseVoice follow-up ran the same lifecycle and execution-control contract
against the pinned Q8 GGUF on macOS arm64 CPU and Metal. Both configurations
completed two load/run/unload cycles, returned monotonic progress, cancelled at
the encoder stage without publishing a result, reused the session afterward,
and serialized two simultaneous calls. The CPU successful/cancelled calls were
210.337/12.3186 ms; Metal measured 41.2061/11.2726 ms. These values are local
behavior observations, not performance gates.

## Open Review Questions

- Whether the generic type names should remain under `runtime` or move into a
  narrower execution-control header before more models adopt them.
- Whether future backends need a capability query distinct from procedure lookup.
- Whether Metal should expose a configurable abort segment count after benchmark
  evidence, rather than using the current fixed split.
- How streaming sessions should bind one control lifetime across start/push/
  finalize operations.
