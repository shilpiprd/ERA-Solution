# Task 
1. You need to have a models folder which contains model for resnet 
2. You need to have a main.py file(my_main.py), and a utils.py file 
3. utils.py file would contain something like image transforms, gradcam, misclassification code, tensorboard related stuff, advanced training polices (like oneCycleLR) etc 
4. Run the model on a ipynb file with minimal code. Use imports as much as possible 
5. show loss curves for train and test 
6. apply gradcam on 10 misclassified images. 
7. train for 20 epochs or so 
8. Apply these transforms while training:
RandomCrop(32, padding=4)
CutOut(16x16)
9. get 10 misclassified images and 10 gradcam images . 

<!-- <p float="left">
  <img src="output_images/image-590.png" width="400" />
  <img src="output_images/image-789.png" width="400" />
  <img src="output_images/image-890.png" width="400" />
  <img src="output_images/image-2039.png" width="400" />
  <img src="output_images/image-2140.png" width="400" />
  <img src="output_images/image-2328.png" width="400" />
  
</p> -->