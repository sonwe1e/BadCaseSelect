# Offline third-party assets

This directory is the only project-local location for vendored source, wheels,
model weights, and their license texts. Runtime code must never download assets.

- `wheelhouse/linux-aarch64/`: wheels produced for the target OS and Python ABI.
- `downloads/`: immutable source archives at the revisions recorded in the manifest.
- `src/`: extracted FLIP, CGVQM, and DINOv2 source snapshots for offline inspection.
- `weights/`: optional scorer weights; current and teacher VFI checkpoints live in `ckpts/`.
- `licenses/`: corresponding license texts.
- `manifest.json`: source, version, license, path, size, and SHA-256 for every distributable external asset.

The prepared manifest contains the pinned source archives, licenses, optional
research weights, a pure project wheel, and manylinux2014 aarch64 NumPy/Pillow
wheels for CPython 3.10, 3.11, and 3.12.  FLIP, CGVQM, and DINOv2 are not
imported by the current mining baseline.  Add the real current/teacher VFI
checkpoints only after the target CANN/Python/torch_npu tuple is known, and
never copy x86_64 wheels into `linux-aarch64`.

Offline archives use the manifest as a closed allow-list for `third_party/`
and `ckpts/`.  The local extracted `src/` trees are inspection workspaces and
are not archived unless their individual files are explicitly recorded.  The
pinned archives in `downloads/` are sufficient to recreate those trees on an
isolated machine.  Unregistered demos, stale wheels, caches, and unrelated
checkpoints are deliberately excluded.

Large payloads under `downloads/`, `src/`, `weights/`, and `wheelhouse/` are
kept out of Git.  Transfer the verified resource-only archive and unpack it at
the repository root after cloning; the manifest remains the audit contract.
