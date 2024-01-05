# Task 
- Get 85% accuracy in test and u can use as many epochs as u want. 
- U must use stride = 2 / (dilation , 200 extra points for this). 
- U can't use MaxPooling. 
- Parameter count must be less than 200k
- U must use depthwise separable convolution 
- U must use 3 blocks and each block should have 3 of 3x3 convolutions and then you're sposed to have an output block , like B1B2B3,OB
- Highly recommended to have a GAP layer and then a 1x1 convolution 
- U must perform the following transforms using albumentations library,
	- horizontal flip
	- shiftScaleRotate
	- coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

### SimpleStructure 
- Here I haven't considered depthwise separable convolution or dilation yet. 
- I've just considered a simple structure with stride and it has 3 transition blocks with k= 3 and stride = 2
- Ive implemented albumentations , one very important point to note is that the A.Normalize() part should happen before any other part.
- In this case there is no Dropout, 
- Params: 186,208 , Test_acc: 85.88 and Train_acc: 80.44


### FullyDepthwiseConv: 
- Here I've implemented dilation in conv 3 and conv6 and implemented depthwise convolution in conv1, 2, 4, 5, 7, 8, 9
- I've added 2% Dropout 
- After implementing DepthWise separable convolution, no. of parameters drastically reduced, so bumped the channels to 128 in some convolutions. 
- Also removed Dilation in conv9( layer just before gap) , this helped . So key_point : never add dilation right before gap layer 
- Also remove DepthWise Dilation from conv1 as it's too close to the input and not performing any specific funciton. So, key_point : Dont use depthwise separable convolution on first convolution layer . 
- Also, in order to add dilation where it was initially strides, the size of the iamge shouldn't change , so padding and stride needed to be adjusted, and hence everywhere where dilation was used, stride = 2 and padding = 2 was set. Formula used for adjusting padding and stride , nout = (nin + 2p - k )/ s + 1 
- - Params: 197,760, Train_acc 79.07 , Test_Acc: 83.77  (for EPOCHS = 50). If epochs increases, acc will easily go above 85. 


KEY TAKEAWAYS: 
- Dont use depthwise separable conv in first layer 
- Dont use dilation in layer just before gap 
- While adding dilation in an originally strided conv, make sure to adjust padding and stride 
- While implementing transforms especially if using Albumentations library, make sure to do A.Normalize() first then perform other albumentations in A.Compose . 

