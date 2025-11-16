import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

minimal_attn = load(name='minimal_attn', sources=['./cpp_file/main.cpp', './cpp_file/flash.cu'], extra_cuda_cflags=['-O2'])

batch_size = 64
n_head = 12
seq_len = 1024
head_embd = 64  # 如果head_embd过大, 则共享内存会不够用 shared memory和 Br,Bc,d 有关

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('results match:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-05))