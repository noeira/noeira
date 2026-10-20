# +--------------------------------------------------------------------------+ #
# | SmolVLA — the frozen half of every observation, computed once per frame
# +--------------------------------------------------------------------------+ #
"""A flat file of the prefix's IMAGE SEGMENT, one row per store row.

Under `train_expert_only` with `state_proj` frozen, everything a frame
contributes to the prefix is a constant of that frame: SigLIP over each
camera, the pixel shuffle, the connector, and the sqrt(960) scale that
`SmolVLAPrefixEmbed.run_images` fuses into the copy. The fine-tune was
recomputing all of it on every visit — and at the reference budget of 20 000
steps x 64 a frame is visited ~65 times. Measured on a 5090 (`SMOLVLA_PROFILE`),
the host image path plus the prefix were **71 ms of a 125 ms observation**.

So this stores that segment: `N_CAM * 64 * 960` fp32 per row — **491 520
bytes**, 9.5 GB for a 19 365-row store. The alternative, caching the KV cache
the prefill produces, is `16 layers x 2 x P x 320` fp32 = 5.7 MB per row,
111 GB for the same store, and would still save nothing on the language and
state tokens. The image segment is the smallest constant and the tower over
it is a few milliseconds at P = 140.

⚠ **EXACT, NOT APPROXIMATE.** The rows are the fp32 numbers `build_prefix`
would have produced, bit for bit — `test_vision_cache.mojo` asserts the
prefix and the prefill output are bit-identical whichever door they came
through. The fine-tune additionally recomputes two rows at startup and
compares them to the file, so a cache built from a different store, a
different checkpoint, or a different `N_CAM` is refused before it trains
anything.

⚠ **NOT VALID once `state_proj` trains.** The image segment does not depend
on `state_proj` — the state is the LAST token, written separately — so the
cache stays exact under `TRAIN_STATE_PROJ`. What changes in that regime is
that the VLM needs its backward, which the cache does not affect. It IS
invalid the moment the vision tower or the connector train, which nothing
here does.

## The file

    bytes [0, 64)     header: 8 x Int64, little-endian
                        magic, version, n_rows, seg, rows_done, 0, 0, 0
    bytes [64, ...)   row i at 64 + i * seg * 4, fp32

`rows_done` is the resume point: rows are written IN ORDER, and the header is
rewritten every `flush_progress`, so a build interrupted at row 12 000 resumes
there rather than at 0. A row above `rows_done` is unreadable by construction
— `read_row` raises rather than returning the zeros `fseeko` past the end
would give.

libc through `external_call`, as `io/proc.mojo` does: the stdlib's file
handle has no seek, and this needs one.
"""

from std.ffi import external_call
from std.os.path import exists

from noeira.nn.core.ptr import mptr


comptime VC_MAGIC: Int64 = 0x5356_4C41_5649_5331
"""'SVLAVIS1' as a big-endian integer — a wrong file is refused at 8 bytes."""
comptime VC_VERSION: Int64 = 1
comptime VC_HEADER_INTS: Int = 8
comptime VC_HEADER_BYTES: Int = VC_HEADER_INTS * 8


struct VisionCache(Movable):
    """One open cache file. `active == False` is the "no cache" object, so a
    caller can hold one unconditionally and branch on the flag."""

    var path: String
    var _fp: Int
    var n_rows: Int
    var seg: Int
    var rows_done: Int
    var active: Bool

    def __init__(out self):
        self.path = String("")
        self._fp = 0
        self.n_rows = 0
        self.seg = 0
        self.rows_done = 0
        self.active = False

    def __init__(out self, *, deinit move: Self):
        self.path = move.path^
        self._fp = move._fp
        self.n_rows = move.n_rows
        self.seg = move.seg
        self.rows_done = move.rows_done
        self.active = move.active

    def __deinit__(deinit self):
        if self.active and self._fp != 0:
            _ = external_call["fclose", Int32](self._fp)

    @staticmethod
    def create(var path: String, n_rows: Int, seg: Int) raises -> Self:
        """A new, empty cache: header written, `rows_done = 0`."""
        if n_rows <= 0 or seg <= 0:
            raise Error("VisionCache.create: n_rows and seg must be positive")
        var mode = String("w+b")
        var fp = external_call["fopen", Int](
            path.as_c_string_span().ptr(), mode.as_c_string_span().ptr()
        )
        if fp == 0:
            raise Error("VisionCache: cannot create " + path)
        var s = Self()
        s.path = path^
        s._fp = fp
        s.n_rows = n_rows
        s.seg = seg
        s.rows_done = 0
        s.active = True
        s.flush_progress()
        return s^

    @staticmethod
    def open(var path: String, n_rows: Int, seg: Int) raises -> Self:
        """An existing cache, checked against the store it is meant for.

        ⚠ Only the SHAPE is checked here. Two stores of the same row count
        and camera count are indistinguishable from the header; the caller
        recomputes a couple of rows and compares — see the fine-tune."""
        if not exists(path):
            raise Error("VisionCache.open: no such file " + path)
        var mode = String("r+b")
        var fp = external_call["fopen", Int](
            path.as_c_string_span().ptr(), mode.as_c_string_span().ptr()
        )
        if fp == 0:
            raise Error("VisionCache: cannot open " + path)
        var hdr = List[Int64](length=VC_HEADER_INTS, fill=Int64(0))
        var got = external_call["fread", Int](
            mptr(hdr), Int(1), VC_HEADER_BYTES, fp
        )
        if got != VC_HEADER_BYTES:
            _ = external_call["fclose", Int32](fp)
            raise Error("VisionCache: " + path + " is shorter than a header")
        if hdr[0] != VC_MAGIC or hdr[1] != VC_VERSION:
            _ = external_call["fclose", Int32](fp)
            raise Error(
                "VisionCache: " + path + " is not a vision cache (bad magic"
                " or version)"
            )
        if Int(hdr[2]) != n_rows or Int(hdr[3]) != seg:
            _ = external_call["fclose", Int32](fp)
            raise Error(
                "VisionCache: " + path + " holds " + String(hdr[2])
                + " rows x " + String(hdr[3]) + " floats, this run needs "
                + String(n_rows) + " x " + String(seg)
                + " — it was built for another store or another camera"
                " count. Delete it, or point SMOLVLA_VISION_CACHE elsewhere."
            )
        var done = Int(hdr[4])
        if done < 0 or done > n_rows:
            _ = external_call["fclose", Int32](fp)
            raise Error(
                "VisionCache: " + path + " reports " + String(done)
                + " rows done of " + String(n_rows) + " — corrupt header"
            )
        var s = Self()
        s.path = path^
        s._fp = fp
        s.n_rows = n_rows
        s.seg = seg
        s.rows_done = done
        s.active = True
        return s^

    def complete(self) -> Bool:
        return self.active and self.rows_done == self.n_rows

    def _seek(self, byte_off: Int) raises:
        if external_call["fseeko", Int32](self._fp, Int64(byte_off), Int32(0)) != 0:
            raise Error(
                "VisionCache: seek to " + String(byte_off) + " failed in "
                + self.path
            )

    def flush_progress(mut self) raises:
        """Rewrite the header with the current `rows_done` and flush.

        Called by the builder every few hundred rows — a build killed between
        two flushes loses those rows and nothing else."""
        if not self.active:
            raise Error("VisionCache: inactive")
        var hdr = List[Int64](length=VC_HEADER_INTS, fill=Int64(0))
        hdr[0] = VC_MAGIC
        hdr[1] = VC_VERSION
        hdr[2] = Int64(self.n_rows)
        hdr[3] = Int64(self.seg)
        hdr[4] = Int64(self.rows_done)
        self._seek(0)
        var n = external_call["fwrite", Int](
            mptr(hdr), Int(1), VC_HEADER_BYTES, self._fp
        )
        if n != VC_HEADER_BYTES:
            raise Error("VisionCache: header write failed in " + self.path)
        if external_call["fflush", Int32](self._fp) != 0:
            raise Error("VisionCache: fflush failed in " + self.path)

    def write_row(mut self, i: Int, mut src: List[Float32]) raises:
        """Append row `i`, which must be the next one: rows are sequential so
        `rows_done` is a resume point and not a bitmap."""
        if not self.active:
            raise Error("VisionCache: inactive")
        if i != self.rows_done:
            raise Error(
                "VisionCache.write_row: rows are written in order — next is "
                + String(self.rows_done) + ", got " + String(i)
            )
        if len(src) != self.seg:
            raise Error(
                "VisionCache.write_row: segment has " + String(len(src))
                + " floats, the cache holds " + String(self.seg)
            )
        self._seek(VC_HEADER_BYTES + i * self.seg * 4)
        var n = external_call["fwrite", Int](
            mptr(src), Int(1), self.seg * 4, self._fp
        )
        if n != self.seg * 4:
            raise Error(
                "VisionCache.write_row: short write at row " + String(i)
                + " in " + self.path + " — disk full?"
            )
        self.rows_done += 1

    def read_row(
        mut self, i: Int, mut dst: List[Float32], dst_off: Int = 0
    ) raises:
        """`dst[dst_off : dst_off + seg]` <- row `i`. Raises past `rows_done`
        rather than reading the zeros an unwritten region would return.
        `dst_off` is how a batch of rows lands back to back in one list."""
        if not self.active:
            raise Error("VisionCache: inactive")
        if i < 0 or i >= self.rows_done:
            raise Error(
                "VisionCache.read_row: row " + String(i) + " is not in the"
                " cache (" + String(self.rows_done) + " of "
                + String(self.n_rows) + " written)"
            )
        if dst_off < 0:
            raise Error("VisionCache.read_row: negative dst_off")
        if len(dst) < dst_off + self.seg:
            dst.resize(dst_off + self.seg, 0.0)
        self._seek(VC_HEADER_BYTES + i * self.seg * 4)
        var n = external_call["fread", Int](
            mptr(dst).unsafe_offset(dst_off), Int(1), self.seg * 4, self._fp
        )
        if n != self.seg * 4:
            raise Error(
                "VisionCache.read_row: short read at row " + String(i)
                + " in " + self.path
            )

    def close(mut self) raises:
        if self.active and self._fp != 0:
            self.flush_progress()
            if external_call["fclose", Int32](self._fp) != 0:
                raise Error("VisionCache: fclose failed for " + self.path)
        self._fp = 0
        self.active = False
