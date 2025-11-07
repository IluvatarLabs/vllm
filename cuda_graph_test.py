import torch
import time

# Pre-allocate tensors before graph capture
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
x = torch.empty(1024, 1024, device='cuda')

# Warmup: initialize cuBLAS before graph capture
x = a @ b
torch.cuda.synchronize()

# Capture two graphs
graph_short = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph_short):
    for _ in range(2):
        x = a @ b

graph_long = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph_long):
    for _ in range(10):
        x = a @ b

# Time replays
start = time.time()
for _ in range(1000):
    graph_short.replay()
torch.cuda.synchronize()
short_time = time.time() - start

start = time.time()
for _ in range(1000):
    graph_long.replay()
torch.cuda.synchronize()
long_time = time.time() - start

print(f"Short graph: {short_time:.3f}s")
print(f"Long graph: {long_time:.3f}s")
print(f"Ratio: {long_time / short_time:.2f}x")  # Should be ~5x