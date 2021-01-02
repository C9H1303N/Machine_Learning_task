import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim


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


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(42),   # 数据增强
        # transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(42),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]),
}

train_dataset = torchvision.datasets.ImageFolder(root='face/train', transform=data_transforms['train'])
test_dataset = torchvision.datasets.ImageFolder(root='face/val', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)


model = AlexNet()
print(model)
model = model.cuda()        # GPU训练
optimizer = torch.optim.Adagrad(model.parameters())     # 优化器 Adagrad
loss_func = torch.nn.CrossEntropyLoss()     # 损失函数交叉熵

for epoch in range(30):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_dataset)), train_acc / (len(train_dataset))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))

torch.save(model.state_dict(), 'model.pth')  # 保存模型
