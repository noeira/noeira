"""Whole-step CUDA-graph capture for the compiled CleanRL benchmarks.

`CudaGraphed(fn)` runs `fn` eagerly for `warmup` calls on a side stream (the
optimizers create their state, torch.compile compiles), captures the next
call into one CUDA graph and replays it from then on. It is PyTorch's
"whole network capture" recipe (torch.cuda.graph docs), the same thing
tensordict's `CudaGraphModule` does for LeanRL:

  * the tensors passed in are copied into static inputs before each replay
    (a CPU tensor becomes one host->device copy);
  * the returned tensors are the graph's static outputs: read them before the
    next call;
  * `fn` must not sync with the host (no `.item()`, no distribution argument
    validation) and must keep its shapes; optimizers need `capturable=True`.
"""

import torch


class CudaGraphed:
    def __init__(self, fn, warmup=3):
        self.fn, self.warmup, self.calls = fn, warmup, 0
        self.graph = None

    def __call__(self, *args):
        if self.graph is None:
            if self.calls < self.warmup:
                self.calls += 1
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    out = self.fn(*[a.to("cuda") for a in args])
                torch.cuda.current_stream().wait_stream(side)
                return out
            self.inputs = [a.to("cuda", copy=True) for a in args]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.fn(*self.inputs)
        else:
            for s, a in zip(self.inputs, args):
                s.copy_(a)
        self.graph.replay()
        return self.outputs
