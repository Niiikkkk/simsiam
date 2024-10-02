import torch

model = torch.load('downloaded/checkpoint_0099.pth.tar', map_location="cpu")
new_state_dict = {}
for name,weights in model["state_dict"].items():
    print(name)
    if name.startswith("module.encoder."):
        name = name.replace("module.encoder.", "")
        new_state_dict[name] = weights
torch.save(new_state_dict, 'resnet50_downloaded.pth')