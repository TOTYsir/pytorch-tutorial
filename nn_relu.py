import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


test = Test()

writer = SummaryWriter("logs_relu")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = test(imgs)
    writer.add_images("output", output, step)

    step += 1

writer.close()
