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
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)
    lrs.append(get_lr(optimizer))                           #adding extra line

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()                                        #adding extra line

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

                                                            #adding get_lr function below
    pbar.set_description(desc= f'Loss={loss.item()} LR={get_lr(optimizer)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    return train_acc, train_losses

def test(model= net, device= device, test_loader= test_loader, criterion= criterion):            #added criterion here
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_images = []  # List to store misclassified images and labels
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim = True)  # Remove keepdim=True; now shape: [batch_size]
            correct += pred.eq(target.view_as(pred)).sum().item()        #modified this line

            # Find misclassified indices
            misclassified_idxs = (pred != target).nonzero(as_tuple=False).squeeze()
            for idx in misclassified_idxs:
                if len(misclassified_images) < 20:  # Collect only 20 images
                    img = data[idx].cpu()
                    actual_label = target[idx].item()
                    predicted_label = pred[idx].item()
                    misclassified_images.append((img, predicted_label, actual_label))
                else:
                    break

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    Accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_acc, test_loss, misclassified_images


# print('printing loss curve and accuracy')
# visualize_loss_accuracy(train_loss=train_losses, train_acc = train_acc, test_loss= test_losses, test_acc = test_acc)
#providing default values