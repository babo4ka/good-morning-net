from GoodMorningClassCNN import GMImagesNet
import torch
from PIL import Image
import torchvision.transforms as transforms


net = torch.load("./resources/goodmrngnet.pt")
net.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = './resources/azaza.jpg'
image = Image.open(path).convert('RGB')
image = image.resize((28, 28))
t = transforms.ToTensor()
img_as_ten = t(image)
x_inp = torch.stack([img_as_ten]).float()
x_inp = x_inp.to(device)
result = net.forward(x_inp)
result = result.argmax(dim=1)
print(result)