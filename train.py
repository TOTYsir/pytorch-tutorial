import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# Prepare for the datasets
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("Length of the train dataset: {}".format(train_data_size))
print("Length of the test dataset: {}".format(test_data_size))

# Load the datasets
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Build up the neural networks
test = Test()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

# Setting up the parameters in the neural networks
# Record the training times, testing times and epoch
total_train_step = 0
total_test_step = 0
epoch = 10

# Add Tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("------------ ROUND {} ---------------".format(i + 1))

    # Training begins
    test.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = test(imgs)
        loss = loss_fn(outputs, targets)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("Training times: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Testing begins
    test.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = test(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("Loss on the test dataset: {}".format(total_test_loss))
    print("Accuracy on the test dataset: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(test, "first_model_{}.pth".format(i))
    print("Model saved")

writer.close()
