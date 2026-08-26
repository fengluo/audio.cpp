# Library-Only Build Mode

## Status

This document describes the proposed `AUDIOCPP_BUILD_APPS` CMake option. The
option is generic to audio.cpp and is intended for downstream projects which
embed `engine_runtime` as a source dependency.

## Motivation

The top-level build currently creates the CLI, HTTP server, GGUF converter,
native model-manager executable, and model performance tool even when a
downstream project only links `engine_runtime`. `EXCLUDE_FROM_ALL` keeps those
targets out of the downstream default build, but still exposes application
targets and runs their configure-time asset generation. This makes the child
build contract broader than the library dependency requires.

## Contract

`AUDIOCPP_BUILD_APPS` defaults to `ON`, preserving the existing standalone build.
When it is `OFF`, CMake does not create:

- `audiocpp_cli`;
- `audiocpp_server`;
- `audiocpp_gguf`;
- `audiocpp_model_manager`;
- `model_perf`.

Server WebUI/demo-voice asset generation and the optional native package-manager
dependency are skipped with the application block. `engine_runtime`, its selected
model closure, backend configuration, examples, and framework tests retain their
own existing controls.

`AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` requires applications because its public
entry points are the standalone manager and managed server. Combining it with
`AUDIOCPP_BUILD_APPS=OFF` is therefore a configuration error instead of silently
ignoring the request.

Application-specific tests are registered only when the corresponding application
targets exist. Framework unit tests remain independently controlled by
`ENGINE_BUILD_TESTS`.

## Downstream Usage

```cmake
set(AUDIOCPP_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ENGINE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(audio.cpp EXCLUDE_FROM_ALL)
target_link_libraries(my_library PRIVATE engine_runtime)
```

The option does not provide an install/export package. That is a separate build
contract and should remain an independently reviewable change.

## Acceptance

- A default standalone configure still creates and builds the existing application
  targets.
- A library-only configure builds `engine_runtime` and the selected model closure.
- The library-only target graph contains none of the five application targets.
- Enabling native model management while applications are disabled fails during
  configuration with an actionable message.
- EchoASR configures audio.cpp as a child with applications disabled and passes its
  CPU and Metal integration suites.
