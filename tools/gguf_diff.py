"""Compare two GGUF files: metadata keys/values, then every tensor's type, shape and bytes.

    python tools/gguf_diff.py a.gguf b.gguf

Byte-identical tensors are counted; differing ones are listed with the share of differing bytes. Used to check that a
pack made on Linux matches the Windows one (docs/LINUX-PLAN.md, Phase 1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py"))
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402


def fields(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            v = f.contents()
        except Exception:
            v = [bytes(p).hex() for p in f.parts]
        out[k] = v
    return out


def main(a_path, b_path):
    a, b = GGUFReader(a_path), GGUFReader(b_path)
    fa, fb = fields(a), fields(b)
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            show = lambda v: (str(v)[:100] + "...") if len(str(v)) > 100 else str(v)
            print(f"meta {k}: {show(fa.get(k, '<missing>'))}  vs  {show(fb.get(k, '<missing>'))}")
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    for n in sorted(set(ta) ^ set(tb)):
        print(f"tensor only in {'a' if n in ta else 'b'}: {n}")
    same = diff = 0
    for n in [t.name for t in a.tensors if t.name in tb]:
        x, y = ta[n], tb[n]
        if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
            print(f"{n}: type/shape {x.tensor_type.name}{list(x.shape)} vs {y.tensor_type.name}{list(y.shape)}")
            diff += 1
            continue
        u, v = np.asarray(x.data).view(np.uint8).ravel(), np.asarray(y.data).view(np.uint8).ravel()
        if np.array_equal(u, v):
            same += 1
        else:
            diff += 1
            print(f"{n}: {x.tensor_type.name} {np.count_nonzero(u != v) / u.size:.2e} of bytes differ")
    print(f"tensors: {same} identical, {diff} different")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
