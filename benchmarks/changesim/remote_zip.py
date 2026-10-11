"""Read single files out of ChangeSim's large public zip archives without downloading them whole.

The archives are shared through anonymous SharePoint links that accept HTTP range requests, so
zipfile can seek inside them. Only the files a sequence needs are fetched.

  python benchmarks/changesim/remote_zip.py list  <part>                 # names + sizes -> index_<part>.txt
  python benchmarks/changesim/remote_zip.py fetch <part> <prefix> <out>  # extract every entry under prefix
"""
from __future__ import annotations

import io
import sys
import time
import urllib.parse
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, Request, build_opener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

PARTS = {   # https://github.com/SAMMiCA/ChangeSim (README, "Download")
    "Query_Seq_Test": "https://kaistackr-my.sharepoint.com/:u:/g/personal/jhyuk_kaist_ac_kr/"
                      "Ecy15_DweZ9EkNdKOFueMn0Bxsq7XkAYNtgHZ-klPZ9M3A?e=5J9Kd3",
    "Ref_Seq_Test": "https://kaistackr-my.sharepoint.com/:u:/g/personal/jhyuk_kaist_ac_kr/"
                    "EQy9p83-nvlMtY90aAnEItkBfJ7-1c3vpAe4dv5xsVlwKA?e=1OAD2l",
}
FILES = {"Query_Seq_Test": "Query_Seq_Test.zip", "Ref_Seq_Test": "Ref_Seq_Test.zip"}
BLOCK = 1 << 20


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small block cache."""

    def __init__(self, part: str):
        self.part = part
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self._authorise()
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        head = self._get(0, 0, raw=True)
        self.size = int(head.headers["Content-Range"].split("/")[-1])

    def _authorise(self) -> None:
        self.opener.open(Request(PARTS[self.part], method="GET")).read()
        path = urllib.parse.quote(f"/personal/jhyuk_kaist_ac_kr/Documents/ChangeSim/{FILES[self.part]}")
        self.url = ("https://kaistackr-my.sharepoint.com/personal/jhyuk_kaist_ac_kr/_layouts/15/download.aspx"
                    f"?SourceUrl={path}")

    def _get(self, start: int, end: int, raw: bool = False):
        for attempt in range(5):
            try:
                resp = self.opener.open(Request(self.url, headers={"Range": f"bytes={start}-{end}"}), timeout=120)
                return resp if raw else resp.read()
            except Exception:                       # expired cookie or a dropped connection
                time.sleep(2 * (attempt + 1))
                self._authorise()
        raise IOError(f"range {start}-{end} failed")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * BLOCK:                            # big reads: straight through
            data = self._get(self.pos, self.pos + n - 1)
        else:
            out = []
            first, last = self.pos // BLOCK, (self.pos + n - 1) // BLOCK
            for b in range(first, last + 1):
                if b not in self.cache:
                    if len(self.cache) > 256:
                        self.cache.clear()
                    self.cache[b] = self._get(b * BLOCK, min(self.size, (b + 1) * BLOCK) - 1)
                out.append(self.cache[b])
            blob = b"".join(out)
            off = self.pos - first * BLOCK
            data = blob[off:off + n]
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("list", "fetch") or (sys.argv[1] == "fetch" and len(sys.argv) < 5):
        raise SystemExit(__doc__)
    cmd, part = sys.argv[1], sys.argv[2]
    zf = zipfile.ZipFile(RangeFile(part))
    if cmd == "list":
        paths.CHANGESIM.mkdir(parents=True, exist_ok=True)
        out = paths.CHANGESIM / f"index_{part}.txt"
        out.write_text("\n".join(f"{i.file_size}\t{i.filename}" for i in zf.infolist()))
        print(f"{len(zf.infolist())} entries -> {out}")
    elif cmd == "fetch":
        prefix, dest = sys.argv[3], Path(sys.argv[4])
        names = [i for i in zf.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
        print(f"{len(names)} files, {sum(i.file_size for i in names) / 1e9:.2f} GB", flush=True)
        started = time.time()
        for k, info in enumerate(sorted(names, key=lambda i: i.header_offset)):
            target = dest / info.filename
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            if k % 500 == 0:
                print(f"  {k}/{len(names)} ({time.time() - started:.0f}s)", flush=True)
        print("done", flush=True)


if __name__ == "__main__":
    main()
