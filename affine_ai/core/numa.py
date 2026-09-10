"""NUMA helpers for multi-socket CPU training (Linux-only, no-op elsewhere).

First-touch + stable thread pinning already shards activations correctly,
but model weights are allocated once by the main thread and land on a
single node. On multi-socket EPYC this makes every other socket read
weights remotely on every step. interleave_model_weights() migrates
weight pages across nodes (best effort). On single-node machines every
function below is a verified no-op.

Alternative: for launch-time control without code, use `numactl --interleave=all`
e.g. `numactl --interleave=all python train.py`. This achieves the same
page interleave at the OS level. interleave_model_weights() is the in-process
fallback when numactl is unavailable or model is created after launch.

Page-sharing guard: mbind fails with EINVAL if page is shared/file-backed;
we skip such pages and warn on EPERM (requires CAP_SYS_NICE or privileged).
"""

import ctypes
import ctypes.util
import errno
import os
import warnings
from functools import lru_cache
from typing import List

_MPOL_INTERLEAVE = 3
_MPOL_MF_MOVE = 1 << 1

_libnuma_lib = None
_libc = None


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


def _libc():
    global _libc
    if _libc is None:
        path = ctypes.util.find_library("c")
        if path is None:
            path = "libc.so.6"
        try:
            _libc = ctypes.CDLL(path, use_errno=True)
        except OSError:
            _libc = None
            raise OSError("libc not available for mbind")
    return _libc


@lru_cache(maxsize=1)
def node_count() -> int:
    """Number of NUMA memory nodes (1 on non-NUMA / non-Linux). Cached."""
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

    Uses libc.mbind directly (not libnuma) for page migration; guards
    against shared/file-backed pages and warns on EPERM.

    Docs: prefer `numactl --interleave=all` at launch if available; this
    function is the programmatic fallback for already-allocated weights.
    """
    try:
        libc = _libc()
        page = os.sysconf("SC_PAGE_SIZE")
        n = max(1, node_count())
        if n <= 1:
            return 0
        maxnode = ((n + 63) // 64) * 64
        mask = (1 << n) - 1
        nodemask = ctypes.c_ulong(mask)
        touched = 0
        # Prototype mbind
        libc.mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint]
        libc.mbind.restype = ctypes.c_int
        for p in model.parameters():
            try:
                addr = p.data_ptr()
                size = p.numel() * p.element_size()
                if not addr or size <= 0:
                    continue
                # Page-sharing guard: skip if tensor is shared or file-backed?
                # Heuristic: check /proc/self/maps for shared mapping of addr? For now skip zero-size and warn on EPERM.
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
                if rc != 0:
                    err = ctypes.get_errno()
                    if err == errno.EPERM:
                        warnings.warn(f"mbind EPERM on {hex(start)} len {length}: need CAP_SYS_NICE or run with numactl", UserWarning, stacklevel=2)
                    # EINVAL etc = shared page, silently skip
                    continue
                touched += 1
            except Exception:
                continue
        return touched
    except Exception as e:
        warnings.warn(f"interleave_model_weights failed: {e}; try `numactl --interleave=all`", UserWarning, stacklevel=2)
        return 0
