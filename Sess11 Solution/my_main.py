from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from transforms import albu_transform, test_transforms

from models.resnet import ResNet18, ResNet34
from tqdm import tqdm 

import numpy as np
from torchvision import datasets, transforms
import cv2
from my_utils import get_lr, visualize_misclassified_images, visualize_loss_accuracy, test_loader, classes, train_loader, device
import torchvision
import matplotlib.pyplot as plt
#for gradcam 
# import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50








lr = 0.1 #(default)
net = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []


def train(model, device= device, train_loader= train_loader, optimizer= optimizer, scheduler= scheduler, criterion= criterion): #adding scheduler and criterion
    model.train()
    pbar = tqdm(train_loader)  #adding
    train_loss = 0
    correct = 0
    total = 0
    # train_acc = []     #adding 
    # train_losses = [] #adding
    processed = 0     #adding
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        train_losses.append(loss)  #adding
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Loss={loss.item()} LR={get_lr(optimizer)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
    return train_acc, train_losses

def test(model, device, test_loader, criterion): 
    model = model.to(device) 
    model.eval() 
    test_loss = 0 
    correct = 0 
    total = 0 
    # test_losses = [] 
    # test_acc = [] 
    with torch.no_grad(): 
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data) 
            loss = criterion(outputs, targets)
            test_loss += loss.item() 
            _, predicted = outputs.max(1) 
            total += targets.size(0) 
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    Accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_acc, test_losses

def visualize_train_data(): 
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # show images
    imshow(torchvision.utils.make_grid(images[:4]))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
