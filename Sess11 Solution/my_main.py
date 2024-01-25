from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from transforms import albu_transform, test_transforms

from models.resnet import ResNet18, ResNet34
from tqdm import tqdm 

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import datasets, transforms
import cv2
from my_utils import get_lr, visualize_misclassified_images, visualize_loss_accuracy
import torchvision
import matplotlib.pyplot as plt
#for gradcam 
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50


cuda = torch.cuda.is_available()
dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train Phase transformations
albu_transforms = A.Compose([ #modified this slightly
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),  # Padding by 4
    A.RandomCrop(32, 32),  # Random Crop to 32x32
    A.HorizontalFlip(p=0.5),  # FlipLR - Horizontal flip
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0, 0, 0), p=0.5),  # CutOut
    ToTensorV2(),
])

def albu_transform(image): 
    # Convert PIL image to numpy array
    image_np = np.array(image)
    # Apply Albumentations transforms
    transformed = albu_transforms(image=image_np)
    return transformed['image']

train = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=albu_transform)
# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)


# Test Phase transformations
test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                       ])
test = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transforms)
# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


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
    train_acc = []     #adding 
    train_losses = [] #adding
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
    test_losses = [] 
    test_acc = [] 
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

def visualize_gradcam_single(model, idx = 4): #pass index of image u wanna perform gradcam on and model.
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    target_layers = [model.layer4[-1]]
    input_tensor = images[idx].unsqueeze(dim=0)# Create an input tensor image for your model..
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    img = input_tensor.squeeze(0).to('cpu').numpy()
    img = np.transpose(img, (1, 2, 0))  # Convert to numpy and reshape to HxWxC
    img = std * img + mean
    img = np.clip(img, 0, 1)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=0.7)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(visualization)

def gradcam_misclassified(model, device): 
   #first get misclassified images
    model.eval()
    misclassified_images = [] 
    actual_labels = [] 
    actual_targets = [] 
    predicted_labels = [] 

    with torch.no_grad(): 
        for data, target in test_loader: #target is the true label
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1) 
            for i in range(len(pred)): 
                if pred[1] != target[i]: 
                        actual_targets.append(target[i]) 
                        misclassified_images.append(data[i])
                        actual_labels.append(classes[target[i]])
                        predicted_labels.append(classes[pred[i]])
    #GRADCAM 
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model= model, target_layers = target_layers)
    #plot the images 
    fig = plt.figure(figsize=(12,5)) #talking about size of each image
    for i in range(10): #we wana plot 10 images
        sub = fig.add_subplot(2, 5, i+1)
        input_tensor = misclassified_images[i].unsqueez(dim = 0)
        targets = [ClassifierOutputTarget(actual_targets[i])] 
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        img = input_tensor.squeeze(0).to('cpu').numpy() 
        img = np.transpose(img, (1,2,0)) 
        img = std* img + mean 
        img = np.clip(img, 0, 1) 

        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=0.7)
        plt.imshow(visualization) 
    plt.tight_layout(visualization)
    plt.show() 