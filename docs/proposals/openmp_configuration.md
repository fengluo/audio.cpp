# Authoritative OpenMP Configuration

## Status

This document describes the proposed OpenMP configuration contract for projects
which build audio.cpp directly or through `add_subdirectory()`.

## Problem

audio.cpp historically exposed `ENGINE_ENABLE_OPENMP`, but did not propagate it
to embedded ggml. A downstream project could disable engine OpenMP while ggml's
independent `GGML_OPENMP` default remained enabled. On Linux this could add an
unexpected `libgomp` dependency and on macOS it could trigger a second OpenMP
probe. Downstreams had to know and set both implementation-layer options.

Several engine and model loops also had unguarded `#pragma omp` directives. They
executed serially when OpenMP was disabled, but GCC could report unknown-pragma
warnings in an otherwise valid non-OpenMP build.

## Contract

`ENGINE_ENABLE_OPENMP` is the audio.cpp-facing source of truth:

- when only `ENGINE_ENABLE_OPENMP` is set, its value is written to
  `GGML_OPENMP` before ggml is configured;
- a preconfigured `GGML_OPENMP` value supplies the compatibility default when
  `ENGINE_ENABLE_OPENMP` has not already been cached;
- if both values are present and disagree, configuration fails with an
  actionable diagnostic instead of silently overriding parent policy;
- enabling OpenMP requires both C and C++ OpenMP components, because ggml has C
  and C++ sources while the engine runtime uses C++;
- disabling OpenMP preserves serial loop behavior and exposes no active OpenMP
  pragmas in audio.cpp sources.

`GGML_OPENMP` remains visible as an implementation cache entry for ggml
compatibility. Downstream projects should set only `ENGINE_ENABLE_OPENMP`.

Changing one value in an existing build directory which already caches both may
produce the intended conflict diagnostic. Set both cached values consistently or
start a clean build directory when changing policy.

## Downstream Usage

```cmake
set(ENGINE_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(audio.cpp EXCLUDE_FROM_ALL)
```

No separate `GGML_OPENMP` assignment is required.

## Acceptance

- A clean `ENGINE_ENABLE_OPENMP=OFF` configure records both engine and ggml
  options as `OFF` without probing for OpenMP.
- A non-OpenMP Qwen3 runtime build succeeds with unknown OpenMP pragmas promoted
  to errors.
- A clean `ENGINE_ENABLE_OPENMP=ON` configure finds the C and C++ components,
  records ggml OpenMP as enabled, builds, links the OpenMP runtime, and executes
  a framework test.
- A configuration with conflicting engine and ggml values fails before ggml is
  added.
- A legacy configuration which sets only `GGML_OPENMP=OFF` produces matching
  engine and ggml values.
- EchoASR sets only `ENGINE_ENABLE_OPENMP=OFF`; clean CPU and Metal child builds
  record `GGML_OPENMP=OFF` and pass their integration suites.

Native Linux x64 validation must additionally confirm that the disabled build
has no `libgomp` dependency and that the enabled build links and executes with
the intended distribution OpenMP runtime.
