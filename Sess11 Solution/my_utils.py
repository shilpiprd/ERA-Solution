from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt 
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam import GradCAM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from my_main import test_loader, classes
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']
  
def visualize_misclassified_images(misclassified_images):
    classes = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    plt.figure(figsize=(10, 10)) #original images were 32x32
    for i, (image, actual, pred) in enumerate(misclassified_images[:10]):
        image = image.numpy().transpose(1, 2, 0)  # Convert to (height, width, channel)
        mean = [0.4914, 0.4822, 0.4465] #3 values for 2 channels, RGB
        std = [0.247, 0.243, 0.261]
        image = image * std + mean  # Undo normalization
        image = np.clip(image, 0, 1)  # Clip values to valid range

        plt.subplot(5, 2, i+1)
        plt.imshow(image)

        plt.title(f"Predicted: {classes[pred]}, Actual: {classes[actual]}")
        plt.axis('off')
    plt.show()

def visualize_loss_accuracy(train_loss, train_acc, test_loss, test_acc): 
    t = [train_items.item() for train_items in train_loss]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

# def visualize_train_data(): 
#    def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

#     dataiter = iter(train_loader)
#     images, labels = next(dataiter)
#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     # show images
#     imshow(torchvision.utils.make_grid(images[:4]))
#     # print labels
#     print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
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

# plot_grad_cam_images(model, test_loader, classes, device)
#call above line to plot gradcam

                        
   