import torchvision.models as models
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):       # 返回当前样本的文件名
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 64, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 7),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = AlexNet()
model.load_state_dict(torch.load('model.pth'))
model.cuda()
model.eval()

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(42),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
}

test_dataset = ImageFolderWithPaths(root='face/test', transform=data_transforms['val'])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

df = pd.read_csv('face/submission.csv')

for pic in test_loader:
    out = model(Variable(pic[0].cuda()))
    pred = torch.max(out, 1)[1].item()      # 预测表情
    # print(pred)
    name = pic[2][0]        # 返回文件路径
    name = name[12:]        # 截取文件名
    # print(name)
    df.loc[df[df['file_name'] == name].index, 'class'] = pred
    # print(df.loc[name, 'class'])

df.loc[df['class'] == 0, ['class']] = 'angry'     # 将编码转换回表情
df.loc[df['class'] == 1, ['class']] = 'disgust'
df.loc[df['class'] == 2, ['class']] = 'fear'
df.loc[df['class'] == 3, ['class']] = 'happy'
df.loc[df['class'] == 4, ['class']] = 'neutral'
df.loc[df['class'] == 5, ['class']] = 'sad'
df.loc[df['class'] == 6, ['class']] = 'surprise'
# print(df)
df.to_csv('face/test/submission.csv', index=False)

