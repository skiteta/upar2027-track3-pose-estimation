"""CUDA-graph replay of a fixed-shape FP16 TorchScript pose forward (deployment helper).

The packaged run.py calls each model once per image with the same view batch
shape, and the RSN forward is bound by host-side kernel launches (~5000 per
call).  Capturing that forward once per shape and replaying it removes the
launch cost without changing the kernels.  This file is embedded verbatim into
the fusion run.py by the packager, so it must stay self-contained.
"""
import torch


def eager_forward(model, inputs):
    """The deployed FP16 forward: autocast without weight cache, TorchScript optimisation off."""
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            return model(inputs).float()


class GraphedForward:
    """One captured CUDA graph per input shape, replayed on copied inputs.

    Any capture error permanently routes that shape to ``eager_forward``; CPU
    inputs always use it.  Graphs keep private memory pools (never shared).
    """

    def __init__(self, model, max_shapes=4):
        self.model, self.max_shapes = model, max_shapes
        self.graphs, self.failed, self.captures = {}, set(), 0

    def _capture(self, inputs):
        static_in = inputs.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager_forward(self.model, static_in)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.jit.optimized_execution(False):
            with torch.cuda.graph(graph):
                with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                    static_out = self.model(static_in).float()
        torch.cuda.synchronize()
        self.captures += 1
        return graph, static_in, static_out

    def __call__(self, inputs):
        with torch.inference_mode():
            key = (tuple(inputs.shape), inputs.dtype)
            if inputs.device.type != 'cuda' or key in self.failed:
                return eager_forward(self.model, inputs)
            entry = self.graphs.get(key)
            if entry is None:
                if len(self.graphs) >= self.max_shapes:
                    return eager_forward(self.model, inputs)
                try:
                    entry = self._capture(inputs)
                except Exception as error:  # capture is an optimisation; never fail the run for it
                    self.failed.add(key)
                    torch.cuda.synchronize()
                    print(f'[submission] CUDA graph capture disabled for {key}: {type(error).__name__}', flush=True)
                    return eager_forward(self.model, inputs)
                self.graphs[key] = entry
            graph, static_in, static_out = entry
            static_in.copy_(inputs)
            graph.replay()
            return static_out.clone()
