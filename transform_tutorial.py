from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/train/ants_image/6240329_72c01e663e.jpg"
img = Image.open(img_path)


writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)

writer.close()

