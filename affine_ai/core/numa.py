"""NUMA helpers for multi-socket CPU training (Linux-only, no-op elsewhere).

First-touch + stable thread pinning already shards activations correctly,
but model weights are allocated once by the main thread and land on a
single node. On multi-socket EPYC this makes every other socket read
weights remotely on every step. interleave_model_weights() migrates
weight pages across nodes (best effort). On single-node machines every
function below is a verified no-op.
"""

import ctypes
import os
from typing import List

_MPOL_INTERLEAVE = 3
_MPOL_MF_MOVE = 1 << 1

_libnuma_lib = None


def _libnuma():
    global _libnuma_lib
    if _libnuma_lib is None:
        for name in ("libnuma.so.1", "libnuma.so"):
            try:
                _libnuma_lib = ctypes.CDLL(name, use_errno=True)
                break
            except OSError:
                continue
    if _libnuma_lib is None:
        raise OSError("libnuma not available")
    return _libnuma_lib


def node_count() -> int:
    """Number of NUMA memory nodes (1 on non-NUMA / non-Linux)."""
    try:
        nodes = os.listdir("/sys/devices/system/node")
        return len([n for n in nodes if n.startswith("node") and n[4:].isdigit()])
    except OSError:
        return 1


def node_cpus(node: int) -> List[int]:
    """CPU ids belonging to a NUMA node ([] if unreadable)."""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            out = []
            for part in f.read().strip().split(","):
                if "-" in part:
                    a, b = part.split("-")
                    out.extend(range(int(a), int(b) + 1))
                elif part:
                    out.append(int(part))
            return out
    except OSError:
        return []


def interleave_model_weights(model) -> int:
    """Migrate every parameter storage to interleave across nodes.

    Best effort: pages that cannot move stay put. Returns number of
    parameters touched. Single-node safe (verifies as no-op).
    """
    try:
        libc = _libnuma()
        page = os.sysconf("SC_PAGE_SIZE")
        n = max(1, node_count())
        maxnode = ((n + 63) // 64) * 64
        mask = (1 << n) - 1
        nodemask = ctypes.c_ulong(mask)
        touched = 0
        for p in model.parameters():
            try:
                addr = p.data_ptr()
                size = p.numel() * p.element_size()
                if not addr or size <= 0:
                    continue
                start = (addr // page) * page
                length = ((addr + size - start + page - 1) // page) * page
                rc = libc.mbind(
                    ctypes.c_void_p(start),
                    ctypes.c_ulong(length),
                    ctypes.c_int(_MPOL_INTERLEAVE),
                    ctypes.byref(nodemask),
                    ctypes.c_ulong(maxnode),
                    ctypes.c_uint(_MPOL_MF_MOVE),
                )
                if rc == 0:
                    touched += 1
            except Exception:
                continue
        return touched
    except Exception:
        return 0
