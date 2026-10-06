# capi_mojo — Mojo calling the MAX C API (no Python at run time)

The smallest Mojo program on the MAX C API: raw `external_call`s that run a CPU vector add from a MEF. For a binding with owned handles, errors raised as Mojo `Error`s, device buffers, capture and the `modular.cfg` fallback built in, see `maxrt/`.

Verified on 2026-10-05 on Linux x86_64 CPU and on 2026-10-06 on an Apple M1, with `max==26.6.0` and `mojo==1.1.0`.

1. **Build the MEF (Python, once).**

   ```bash
   python noeira_max/capi_mojo/build_mef.py      # writes graph.mef (CPU, a + b, symbolic dim "n")
   ```

2. **Build the Mojo executor.** Use `mojo build`, because `mojo run` cannot resolve the `libmax` symbols.

   ```bash
   LIB=$CONDA_PREFIX/lib        # pip wheels: site-packages/modular/lib
   mojo build noeira_max/capi_mojo/run_mef.mojo -o /tmp/run_mef \
     -Xlinker -L$LIB -Xlinker -lmax -Xlinker -rpath -Xlinker $LIB
   ```

3. **Run it** from the directory that contains `graph.mef`:

   ```bash
   /tmp/run_mef
   ```

   The expected output is `[9.0, 9.0, ...]` plus the per-call `M_executeModelSync` time.

**Runtime configuration.** The runtime finds `libKGENCompilerRTShared` through `$MODULAR_HOME/modular.cfg`.

- Inside `pixi run`, activation sets it, so nothing is needed.
- Outside an activated environment, `M_initModel` fails with `unable to locate compiler_rt /lib/libKGENCompilerRTShared.so`. To fix it, write a `modular.cfg` with:
  - `[max] package_root=<root>`;
  - `[mojo-max] compilerrt_path=<root>/lib/libKGENCompilerRTShared.so` and `mgprt_path=<root>/lib/libMGPRT.so`.

  Then point `MODULAR_HOME` at the directory that contains it.

On CPU, the floor is ~46 µs per `M_executeModelSync` call on a cloud x86 container and 70–100 µs on an Apple M1 laptop under load (8-element add). On the GPU, see `bench/`.

**macOS:** `libmax.dylib` exports the C API too: its 70 `M_*` functions are in the Mach-O export trie, which `grep` on the binary (and so `../probe_c_api.sh`) misses.
