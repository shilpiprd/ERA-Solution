# Task 
1. Convert entire base code into pytorch lightning 
2. create a space on huggingface and ask the user to upload images, display all classes ur model classified, and must incorporate gradcam. 
3. Train the model to reach such that all of these are true:
    Class accuracy is more than 75%
    No Obj accuracy of more than 95%
    Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
    Ideally trailed till 40 epochs 
4. Add these training features:
    Add multi-resolution training - the code shared trains only on one resolution 416
    Add Implement Mosaic Augmentation only 75% of the times
    Train on float16
    GradCam must be implemented.
5. Requirement for space app: 
    1. It should allow upload of images from user. 
    2. show output of uploaded images as well as sample iamges 
    3. show gradcam on images uploaded by user as well as on sample 
    4. show htings like the classes that ur model predicts as well as link to ur actual model.

## Solution: 
1. data was downloaded from kaggle using kaggle api since size(data) = 5gb is too big to be uploaded to github. 
2. model saves chekcpoint after each epoch in gdrive so that next time training could start from that epoch. 
3. converting all code from main.py to pytorch lightning (implementation in my_pl_detection.py) was the most difficult part. 
4. implementing a gradio interface was also very difficult. 
5. gradcam.py contains all the required gradcam functions. 
6. For 75% mosaic implementatino, it couldn't be implmented at time of trianing. but code for it has been written now. 
