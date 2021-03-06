import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from ResNet import *
from tqdm import tqdm

def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight.data)
        
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet(BasicBlock, 3, len(classes))  # ResNet-20
# net = ResNet(BasicBlock, 9, len(classes))  # ResNet-56
# net = ResNet(BasicBlock, 18, len(classes)) # ResNet-110
# NET = ResNet(PreActBlock, 3, len(classes))  # ResNet-20-preact
# net = ResNet(PreActBlock, 9, len(classes))  # ResNet-56-preact
# net = ResNet(PreActBlock, 18, len(classes)) # ResNet-110-preact

net = net.to(device)

if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[81,122], gamma=0.1)
criterion = nn.CrossEntropyLoss()
net.apply(weights_init)

result = {
    'train-loss': [],
    'train-acc': [],
    'test-loss': [],
    'test-acc': [],
}

def train(epoch):
    global result
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    progress = tqdm(enumerate(trainloader), total=len(trainloader), ascii=True)
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        progress.set_description('Loss: %.6f | Acc: %.3f%% (%d/%d) | LR: %g'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))

    result['train-loss'].append(train_loss/(batch_idx+1))
    result['train-acc'].append(1.*correct/total)

def test(epoch):
    global result
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    progress = tqdm(enumerate(testloader), total=len(testloader), ascii=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            progress.set_description('Loss: %.6f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        result['test-loss'].append(test_loss/(batch_idx+1))
        result['test-acc'].append(1.*correct/total)
    
for epoch in tqdm(range(164), desc='Epoch', ascii=True):
    scheduler.step()
    train(epoch)
    test(epoch)

with open('csv/resnet-20-test.csv', 'w') as f:
    f.write('epoch,train-loss,train-acc,test-loss,test-acc\n')
    for idx, arr in enumerate(zip(result['train-loss'], result['train-acc'], result['test-loss'], result['test-acc'])):
        f.write('%d,%s\n' % (idx, ','.join(map(str, arr))))
