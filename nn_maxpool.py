import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool_1(input)
        return output


test = Test()
step = 0
writer = SummaryWriter("logs_maxpool")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = test(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
