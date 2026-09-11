"""Byte-level recall evals: copy / phone-book / needle-in-haystack.

Teacher-forced scoring (no generation loop): feed the full sequence, measure
next-byte prediction over the answer span. Deterministic, fast, GPU-light.
Usage: evaluate_recall(model, device) -> {task: {span_acc, byte_acc}}.
"""

import random
import torch


def _bytes(s: str):
    return list(s.encode("latin-1"))


def make_copy(rng, span_len=32):
    span = [rng.randrange(97, 123) for _ in range(span_len)]
    seq = span + _bytes("|") + span
    return seq, list(range(span_len + 1, span_len + 1 + span_len))


def make_phonebook(rng, n_pairs=8):
    pairs = []
    for _ in range(n_pairs):
        name = "".join(chr(rng.randrange(97, 123)) for _ in range(5))
        num = f"{rng.randrange(100, 1000):03d}"
        pairs.append((name, num))
    seq = []
    for name, num in pairs:
        seq += _bytes(f"{name}:{num};")
    qname, qnum = pairs[rng.randrange(n_pairs)]
    seq += _bytes(f"{qname}:")
    ans = _bytes(qnum)
    seq += ans
    lo = len(seq) - len(ans)
    return seq, list(range(lo, len(seq)))


def make_niah(rng, total_len=256):
    num = f"{rng.randrange(10000, 99999):05d}"
    needle = _bytes(f"The magic number is {num}.")
    fill = [rng.randrange(97, 123) for _ in range(total_len - len(needle) - 20)]
    pos = rng.randrange(0, max(1, len(fill)))
    seq = fill[:pos] + needle + fill[pos:] + _bytes("Magic:")
    ans = _bytes(f" {num}")
    seq += ans
    lo = len(seq) - len(ans)
    return seq, list(range(lo, len(seq)))


def _pad_batch(seqs, pad=0):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [pad] * (n - len(s)) for s in seqs], dtype=torch.long)


@torch.no_grad()
def evaluate_recall(model, device="cuda", n_samples=48, seed=0):
    rng = random.Random(seed)
    makers = {"copy": lambda: make_copy(rng),
              "phonebook": lambda: make_phonebook(rng),
              "niah": lambda: make_niah(rng)}
    out = {}
    hyb = getattr(model, "hybrid", model)
    was_training = hyb.training
    hyb.eval()
    for name, mk in makers.items():
        seqs, spans = zip(*[mk() for _ in range(n_samples)])
        x = _pad_batch(seqs).to(device)
        res = hyb(x)
        logits = res[0] if isinstance(res, (tuple, list)) else res
        pred = logits.argmax(dim=-1)
        span_ok, byte_ok, byte_tot = 0, 0, 0
        for b, sp in enumerate(spans):
            ok = True
            for t in sp:
                if t >= x.shape[1]:
                    continue
                hit = bool(pred[b, t - 1] == x[b, t]) if t > 0 else False
                byte_tot += 1
                byte_ok += hit
                ok = ok and hit
            span_ok += ok
        out[name] = {"span_acc": span_ok / n_samples, "byte_acc": byte_ok / max(1, byte_tot)}
    if was_training:
        hyb.train()
    return out


def make_recall_corpus(n_docs=4000, seed=1):
    rng = random.Random(seed)
    makers = [lambda: make_copy(rng), lambda: make_phonebook(rng), lambda: make_niah(rng)]
    blob = bytearray()
    for i in range(n_docs):
        s, _ = makers[i % 3]()
        blob.extend(bytes(s))
    return blob
