import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import datasets, transforms
import cv2 
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

# Test Phase transformations
test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                       ])

                            
