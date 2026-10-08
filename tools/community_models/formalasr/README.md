# FormalASR conversion recipe

FormalASR-0.6B uses the existing `qwen3_asr` family. audio.cpp owns conversion
and format compatibility; EchoASR validates its public inference API; EchoX
Desktop owns dictation evaluation, distribution URLs and installation.

## Historical artifact reproduction

This recipe reproduces the October 6, 2026 Q4/Q8 artifacts. It pins the original
converter, source checkpoint and sidecars, model spec and tied-embedding patch.
The archived spec differs from the current spec. `legacy-tied-embedding.patch`
is historical evidence, not a patch to apply again to the current checkout.
The old converter embeds the canonical source directory basename in
`general.name`; the leaf directory must remain named `source` for byte-identical
reproduction. Only recipe text belongs in Git. Weights, GGUFs and executables live outside
all three source repositories:

```text
../model-artifacts/formalasr-0.6b/
  source/702f3feb23d0ee8a327ba2d5882d2ecf1517d7f3/source/
  audiocpp-q4-v1/
    formal-q4_k.gguf
    manifest.json
    conversion.json / conversion.log
    LICENSE.txt / NOTICE.txt
  audiocpp-q8-v1/                   # Same layout, Q8 output
  history/2026-10-06/
    bin/                          # Original macOS arm64 converter and CLI
    evidence/                     # Immutable original logs and commands
```

From the audio.cpp repository root:

```sh
python3 tools/community_models/formalasr/manage.py verify
python3 tools/community_models/formalasr/manage.py convert \
  --type q4_k --output build/formalasr-reproduction/formal-q4_k.gguf
```

Use `--artifacts-root /path/to/formalasr-0.6b` before the subcommand to select
another storage root. An output, log or record already present is never
overwritten. Input hashes are verified before conversion; output size and hash
must match the selected historical artifact before atomic installation.
Conversion records and logs are saved next to the new GGUF.

Migration validation on October 8: all 28 inventory files passed; fresh Q4
conversion reproduced 854,796,384 bytes and SHA-256
`fa5b216030d3fc4a316cbe8d90d17f58ed0f8660f8d3554b4778581e3ae5d1ba`.
Logs and the migration summary are in the shared `history/migration-2026-10-08/`
directory. Duplicate test GGUFs were removed after comparison. Old experiment
source/Q4/Q8 paths are compatibility symlinks, not separately managed assets.

`inventory.json` covers the original 28 files and `provenance.json` records
source/converter/spec identities. Source: `TaurenMountain/FormalASR-0.6B` revision
`702f3feb23d0ee8a327ba2d5882d2ecf1517d7f3`; Apache-2.0 declaration.
Original converter: audio.cpp `72c84209384204c75f6151cb44ea680a414fe7c9` plus the
archived patch. The binary is macOS arm64 only; rebuilding for another platform
requires validation. It is not a production inference dependency.

## New conversions

Use the current `audiocpp_gguf` and current `model_specs/qwen3_asr.json` for a
new recipe. Write to a new artifact version directory and record source revision,
converter commit/build flags, spec hash and output hash. Do not overwrite v1
or assume a newer converter produces its historical hash. Validate numeric,
inference and dictation quality separately before changing a deployment pin.
The shared directory is local storage, not a published model service.

## Consumers

EchoASR tests accept the shared GGUF via `ECHOASR_FORMAL_MODEL_PATH`.
EchoX Desktop imports a pinned copy of `manifest.json` and license/notice files
using `scripts/sync-formal-artifact.py`. Its runtime remains independent of this
workspace and downloads/installs the model under Application Support.
Historical dictation evidence remains in EchoX Desktop, with original hashes
and paths retained as evidence. General GGUF conversion is documented in
[`docs/gguf.md`](../../../docs/gguf.md).
