import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image

# load model

class FashionCNN(nn.Module):
  def __init__(self, in_channels=1, num_classes=10):
    super(FashionCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1) # same convolution
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc = nn.Linear(16*7*7, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)

    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)

    return x

in_channels = 1
num_classes = 10
model = FashionCNN(in_channels=in_channels, num_classes=num_classes)
PATH = "fashion_with_cnn.pth"
model.load_state_dict(torch.load(PATH))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    images = image_tensor
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
