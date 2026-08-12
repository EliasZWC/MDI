"""
mdi_version.py - single source of truth for the MDI project version and the
current canonical weight file(s).

RATIONALE (user directive 2026-08-13): "任何实验都得记录版本，版本号不相符就不能
随便引用" — every experiment must record its version, and you may only cite
results whose version matches. To keep this manageable, the version + the
current weight filenames live HERE (one place), and every experiment script
imports them and stamps its log header with them. When a new version/weight
is released you update this file ONLY; no per-script edits.

Usage in any experiment script:
    from mdi_version import VERSION, W_MDI, W_MDI_MPNET
    ... log.write(f"version={VERSION} W={W_MDI} W-mpnet={W_MDI_MPNET}\n")
"""

# ---------------------------------------------------------------------------
# Project semantic version (mirror README **Version:** and git tag).
# Increment per semantic rules: fix +0.0.1, feature +0.1.0, breaking +1.0.0.
# ---------------------------------------------------------------------------
VERSION = "0.3.1"

# ---------------------------------------------------------------------------
# Current canonical weight files (mapping step, MDI-phi).
# Scripts should default to these; update here when a new weight is canonical.
#   W_MDI       -> phi projection for MiniLM base (384-d -> 64-d)
#   W_MDI_MPNET -> phi projection for mpnet  base (768-d -> 64-d)
# NOTE: the two base models have DIFFERENT input dimensions, so these MUST
# stay distinct — never point both at the same file (matmul dimension crash).
# ---------------------------------------------------------------------------
W_MDI = "mdi_W.npy"              # MiniLM v1 (384,64) — current minilm phi
W_MDI_MPNET = "mdi_W_v2b_mpnet.npy"  # mpnet v2b (768,64) — current mpnet phi


def weight_stamp(path):
    """Return a short human-readable provenance stamp for a weight file, or
    'NONE' if the file does not exist. Use in every log header."""
    import datetime
    import os
    if not path or not os.path.exists(path):
        return f"{path}=MISSING"
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime(
        "%Y-%m-%d %H:%M:%S")
    return f"{path} (mtime {mt})"


def header(extra=""):
    """Standard version banner for experiment log/print headers."""
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 72,
        f"MDI version={VERSION}",
        f"W_MDI      = {weight_stamp(W_MDI)}",
        f"W_MDI_MPNET= {weight_stamp(W_MDI_MPNET)}",
        f"run-time   = {now}",
    ]
    if extra:
        lines.append(extra)
    lines.append("=" * 72)
    return "\n".join(lines)
