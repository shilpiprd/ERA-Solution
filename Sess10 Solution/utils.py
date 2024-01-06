from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']
  