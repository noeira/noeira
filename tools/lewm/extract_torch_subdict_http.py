"""Pull ONE sub-dict of a remote PyTorch checkpoint by HTTP range reads.

    python extract_torch_subdict_http.py URL KEY OUT.pt [--bf16]

A `torch.save` file is an uncompressed zip: `<prefix>/data.pkl` (the pickled
structure, tensors as persistent ids) plus one `<prefix>/data/<key>` record
per storage. This reads the pickle with a lazy unpickler, keeps `ckpt[KEY]`,
and fetches only the storages its tensors use. Written for V-JEPA 2.1
ViT-G/384 (`vjepa2_1_vitG_384.pt`, 30 GB with the predictor and the
optimizer state; its `target_encoder` alone is ~7 GB in fp32), on a box
whose disk cannot hold the whole file. Saves `{KEY: state_dict}`.
"""
import argparse
import io
import pickle
import sys
import time
import zipfile

import requests
import torch

CHUNK = 32 << 20
RETRIES = 12


class _RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.s = url, 0, requests.Session()
        h = self.s.head(url, allow_redirects=True)
        h.raise_for_status()
        self.size = int(h.headers["content-length"])
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        # short reads of CHUNK at most (BufferedReader.read loops until it has
        # n), each retried: a 7 GB pull over one connection does break
        n = min(len(b), self.size - self.pos, CHUNK)
        if n <= 0:
            return 0
        for attempt in range(RETRIES):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
                r.raise_for_status()
                data = r.content
                if len(data) == n:
                    break
            except requests.RequestException as e:
                print(f"  range {self.pos}+{n}: {type(e).__name__}, retry {attempt + 1}", flush=True)
            time.sleep(min(2 ** attempt, 30))
        else:
            raise IOError(f"range {self.pos}+{n}: {RETRIES} attempts failed")
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


class _Lazy:
    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


class _LazyTensor:
    def __init__(self, st, off, size, stride):
        self.st, self.off, self.size, self.stride = st, off, tuple(size), tuple(stride)


def _rebuild(storage, offset, size, stride, *rest):
    return _LazyTensor(storage, offset, size, stride)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.state = s


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        try:
            return super().find_class(mod, name)
        except Exception:
            return _Dummy

    def persistent_load(self, pid):
        _, stype, key, _, numel = pid
        dtype = stype if isinstance(stype, torch.dtype) else getattr(stype, "dtype", torch.float32)
        return _Lazy(key, dtype, numel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("key")
    p.add_argument("out")
    p.add_argument("--bf16", action="store_true")
    a = p.parse_args()
    raw = _RangeFile(a.url)
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    ck = _Unpickler(io.BytesIO(zf.read(pkl))).load()
    sub = ck[a.key]
    print(f"{a.url}: {raw.size / 1e9:.1f} GB, keys {list(ck)[:8]}; {a.key}: {len(sub)} tensors", flush=True)
    cache, out = {}, {}
    for i, (name, lt) in enumerate(sub.items()):
        st = lt.st
        if st.key not in cache:
            buf = bytearray(zf.read(f"{prefix}data/{st.key}"))
            cache = {st.key: torch.frombuffer(buf, dtype=st.dtype)}  # one storage at a time
        t = cache[st.key].as_strided(lt.size, lt.stride, lt.off).clone()
        out[name.replace("module.backbone.", "")] = t.bfloat16() if a.bf16 and t.is_floating_point() else t
        if i % 100 == 0:
            print(f"  {i}/{len(sub)} | fetched {raw.fetched / 1e9:.2f} GB", flush=True)
    n = sum(t.numel() for t in out.values())
    torch.save({a.key: out}, a.out)
    print(f"saved {a.out}: {len(out)} tensors, {n / 1e9:.3f} B params, fetched {raw.fetched / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
