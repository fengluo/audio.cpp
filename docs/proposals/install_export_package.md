# Installable CMake Runtime Package

## Status

This proposal defines the package contract for downstream projects which cannot
or should not embed audio.cpp with `add_subdirectory()`. It deliberately follows
the library-only build proposal: the two contracts solve different problems and
must remain independently reviewable.

The local candidate adds the install/export rule, `AudioCppConfig.cmake`,
and the downstream fixture described below. A macOS arm64 CPU configuration with
`AUDIOCPP_BUILD_APPS=OFF`, `AUDIOCPP_MODEL_SET=core`, portable CPU settings, and
OpenMP disabled passed the fixture: it installed to a disposable prefix, a fresh
consumer found only that prefix, linked `AudioCpp::engine_runtime`, and reported
the mandatory `marblenet_vad,silero_vad` registry. The fixture also rejects
source/build paths in the exported targets file.

The same fixture also passed on macOS arm64 with Metal enabled; its fresh
consumer linked the exported Metal closure without naming a ggml target or a
framework. Linux still needs the same fixture evidence before this is presented
as an accepted upstream package contract.

## Goal

A configured and installed audio.cpp tree must be consumable from a clean,
unrelated source directory with:

```cmake
find_package(AudioCpp CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE AudioCpp::engine_runtime)
```

The consumer must not add the audio.cpp source tree, name ggml backend targets,
or reproduce link-order and platform-framework details. The package is a C++17
runtime package, not an application distribution and not a model downloader.

## Public package surface

The installed CMake package is named `AudioCpp` and exports exactly one
downstream entry point:

| Imported target | Purpose |
|---|---|
| `AudioCpp::engine_runtime` | Selected model-runtime closure and the headers under `include/engine/` |

The target remains static or shared according to the producer's normal
`BUILD_SHARED_LIBS` choice. It exposes the same selected model composite that
was configured by `AUDIOCPP_MODEL_SET` and `AUDIOCPP_MODELS`; it does not make
unselected providers appear at link time.

The package config records the selected, linked family list as read-only package
metadata. That metadata is diagnostic only. The runtime capability registry
remains the authoritative answer for an executable, because backend discovery is
process and platform dependent.

The installed public header tree is `include/engine/`. Its ggml header
requirements are carried by the imported target's usage requirements; consumers
must not add an include path for ggml themselves.

## Closure and backend policy

`engine_runtime` is not a self-sufficient archive. In a static build its object
code requires the selected ggml targets, SentencePiece, libyaml, and cJSON.
ggml itself may require CPU, Metal, CUDA, HIP, Vulkan, OpenMP, thread, math, or
dynamic-loader libraries. Exporting only `engine_runtime` would therefore create
a package that configures successfully but fails to link a normal consumer.

The implementation must export or bundle the complete configured closure and
make it an implementation detail of `AudioCpp::engine_runtime`. In particular:

- ggml base plus every target enabled by the chosen backend configuration are
  part of the export set;
- SentencePiece, libyaml, and cJSON are included in the closure with correct
  static-link order;
- required CMake dependencies, such as `Threads` and OpenMP when enabled, are
  resolved by `AudioCppConfig.cmake` before imported targets are loaded;
- platform frameworks and accelerator SDK runtimes remain producer-selected
  requirements, rather than becoming consumer CMake options;
- no exported usage requirement may contain an audio.cpp build-directory or
  source-directory path.

audio.cpp's vendored ggml and SentencePiece projects currently carry their own
install rules. The package implementation must prevent those rules from
installing standalone applications or overlapping package files when audio.cpp
is installed. AudioCpp owns the install manifest for this closure.

## Models and runtime data

The compiled closure establishes which providers can register; it does not ship
model weights. Weights, licenses, model-cache placement, download policy, and
application privacy decisions stay outside this package.

Model specs need a separate rule from headers and libraries. A standalone GGUF
package embeds its model spec and is immediately usable by an installed runtime.
For a non-embedded or legacy artifact, the application supplies a model-scoped
`model_spec_override`, or places the compatible `model_specs/<family>.json`
beside the model as documented by the runtime. A static library cannot reliably
discover the prefix from which its final application was installed, so an
install prefix must not become an implicit runtime model-spec search root.

`AUDIOCPP_DEPLOYMENT_BUILD=ON` remains the explicit opt-in for compiling the
catalog into the runtime. The normal package must not silently turn it on,
because doing so changes model-spec update and release semantics.

## Exclusions

The package does not install or export:

- `audiocpp_cli`, `audiocpp_server`, `audiocpp_gguf`, model-management tools, or
  performance utilities;
- a package-manager network client or model catalog downloader;
- model weights, generated model caches, test fixtures, or WebUI assets;
- raw build-tree targets as a second public API.

Applications remain governed by `AUDIOCPP_BUILD_APPS`; a release that wants
executables packages them through its own distribution flow.

## Required fixture

The implementation adds a source-independent CTest fixture under
`tests/install_package/` with this sequence:

1. configure audio.cpp with `AUDIOCPP_BUILD_APPS=OFF`, an explicit custom model
   composite, and one portable backend policy;
2. build and install into a disposable prefix;
3. configure a separate consumer source tree with only
   `CMAKE_PREFIX_PATH=<prefix>`;
4. have that consumer call `find_package(AudioCpp CONFIG REQUIRED)`, link only
   `AudioCpp::engine_runtime`, include a public engine header, and create the
   default registry;
5. build and execute it, asserting that its registered families equal the
   configured selected closure plus the mandatory VAD loaders;
6. inspect the consumer's link closure and fail when it references the producer
   source or build directory.

The fixture contains no `add_subdirectory`, raw library paths, copied headers,
or model weights. A Linux run must additionally check the configured OpenMP
policy and dynamic runtime dependencies. macOS CPU and Metal runs must verify
their respective backend capability results.

## Acceptance criteria

- A default standalone build remains unchanged when it is not installed.
- A library-only install exports one documented target and no application
  targets.
- A clean external consumer configures, builds, links, and runs using only the
  install prefix and `find_package(AudioCpp CONFIG REQUIRED)`.
- The exported target closes every selected static dependency and backend
  requirement without consumer-side target names or link-order workarounds.
- The installed package does not embed a source/build path in public include or
  link requirements.
- The custom-model fixture reports exactly the configured provider closure.
- A deployment build and a non-deployment build retain their distinct model-spec
  behavior.
- CPU and Metal acceptance runs pass before the interface is presented as a
  supported downstream package; a Linux package-fixture run and native Linux
  x64 model inference remain separate acceptance gates.
