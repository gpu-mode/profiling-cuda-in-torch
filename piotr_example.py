import torch
torch.cuda.cudart().cudaProfilerStart()
torch.square(torch.randn(10000, 10000).cuda())
torch.cuda.cudart().cudaProfilerStop()