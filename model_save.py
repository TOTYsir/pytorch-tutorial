import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

torch.save(vgg16, "vgg16_method1.pth")

# Saave parameters (Official)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

