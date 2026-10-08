#!/usr/bin/env python3
"""Verify archived FormalASR assets or reproduce their offline GGUF conversion."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


RECIPE = Path(__file__).resolve().parent
INVENTORY = json.loads((RECIPE / "inventory.json").read_text())
DEFAULT_ARTIFACTS = RECIPE.parents[3] / "model-artifacts" / "formalasr-0.6b"
ARTIFACTS = DEFAULT_ARTIFACTS


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(path, expected):
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise RuntimeError(f"Size or SHA-256 mismatch: {path}")


def verify():
    for entry in INVENTORY["files"]:
        root = ARTIFACTS if entry["root"] == "artifacts" else RECIPE
        check(root / entry["path"], entry)
    print(json.dumps({"verified_files": len(INVENTORY["files"]),
                      "artifacts": str(ARTIFACTS)}))


def convert(args):
    # Every sidecar contributes to the embedded GGUF; verify all pinned inputs.
    for entry in INVENTORY["files"]:
        if entry["root"] == "artifacts" and (
            entry["path"].startswith("source/") or entry["path"] == "history/2026-10-06/bin/audiocpp_gguf"
        ):
            check(ARTIFACTS / entry["path"], entry)
        elif entry["root"] == "recipe" and entry["path"] == "qwen3_asr.json":
            check(RECIPE / entry["path"], entry)

    output = args.output.expanduser().resolve()
    log = output.with_name(output.name + ".conversion.log")
    record = output.with_name(output.name + ".conversion.json")
    if any(path.exists() for path in (output, log, record)):
        raise RuntimeError("Output or conversion record exists; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.part")
    command = [str(ARTIFACTS / "history/2026-10-06/bin/audiocpp_gguf"),
               "--input", str(ARTIFACTS / "source/702f3feb23d0ee8a327ba2d5882d2ecf1517d7f3/source/model.safetensors"),
               "--root", str(ARTIFACTS / "source/702f3feb23d0ee8a327ba2d5882d2ecf1517d7f3/source"),
               "--output", str(temporary), "--type", args.type,
               "--family", "qwen3_asr",
               "--model-spec", str(RECIPE / "qwen3_asr.json")]
    details = {"command": command, "output": str(output), "status": "started"}
    record.write_text(json.dumps(details, indent=2) + "\n")
    try:
        with log.open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        expected = next(entry for entry in INVENTORY["files"]
                        if entry["root"] == "artifacts"
                        and entry["path"] == f"{'audiocpp-q4-v1' if args.type == 'q4_k' else 'audiocpp-q8-v1'}/formal-{args.type}.gguf")
        check(temporary, expected)
        # Publish a verified file atomically without overwriting an existing result.
        os.link(temporary, output)
        details.update(status="verified", bytes=expected["bytes"], sha256=expected["sha256"])
    except Exception as error:
        details.update(status="failed", error=str(error))
        raise
    finally:
        record.write_text(json.dumps(details, indent=2) + "\n")
        temporary.unlink(missing_ok=True)
    print(json.dumps(details, indent=2))


def main():
    global ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS,
                        help="Shared FormalASR asset directory outside source repositories")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("verify", help="Check sizes and SHA-256 for all archived files")
    conversion = commands.add_parser("convert", help="Reproduce a pinned GGUF offline")
    conversion.add_argument("--type", choices=("q4_k", "q8_0"), default="q4_k")
    conversion.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ARTIFACTS = args.artifacts_root.expanduser().resolve()
    try:
        verify() if args.action == "verify" else convert(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
