from torchvision import transforms , datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize(train_loader): 
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(12): #if you've set shuffle = False, then every time u run this cell, u'll get same output.
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])


# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1407,), (0.4081,))
    transforms.Normalize((0.1307,), (0.3081,)) #both training and test data should've same
    ])

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, transform=test_transforms)  #changed this

train_losses = []
test_losses = []
train_acc = []
test_acc = []