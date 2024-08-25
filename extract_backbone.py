import torch

model = torch.load('/home/nberardo/simsiam/checkpoint_0099.pth.tar', map_location="cpu")
for name in model:
    print(name)