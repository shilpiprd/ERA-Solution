# Task 
Here I'll explain different skeletons I took, and the way I arrived at the final result 


### Skeleton.ipynb 
- Here I've considered ReLU after each layer and a dropout of 10%. 
- Target: 
- 	With reduced parameters, reduce overfitting 
- Obsv: 
-	Test acc. increased propelry as compared to with no Dropout, ReLU. 
-	We see much less overfitting 
- 	One thing we obsvd for sure is that Dropout layer is effective in dealing with Overfitting. 
-	Some experiment needs to be done with 15% Dropout. 
- 	Also, we know that accuracy increases with Batch Normalization, so we'll add that next as well. 
- 	Best ``Train_acc`` = 99.06 and ``Test_acc`` = 99.14   & ``Params`` = 18,132. 


### WIP_1.ipynb 
- Target (Changes Made): 
- 	Noticed that 15% Dropout gave better accuracy for both train and test. 
- 	Then added Batch Normalization which further improved accuracy. 
- Obsv.:
- 	It's better as compared to Batch Normalization and Dropout =10%. 
- 	Model still shows overfitting. 
- 	Best ``train_acc`` = 99.33 and ``test_acc`` = 99.11 & ``Params`` = 18,312
-	Next step is to try GAP while making sure that model capacity doesn't decrease by adding an extra layer after gap. 


### WIP_2.ipynb
- Target (Changes Made): 
	- We've 15% Dropout, BN, added GAP with 17872 parameters. Results weren't very good. 
	- Added an extra layer after GAP and changed the architecture slightly to make sure Receptive Field reaches 28. 
- Obsv: 
	- This layer and slight architecture modification made good change. 
	- Channel Movement: 1 > 16 > 32 > 22 > 10 > 32 > 32 > 16> 10 
	- RF movement: 1 > 3 > 5> 6 > 10 > 14 > 28 
	- Best ``Train_acc`` = 99.30 and ``Test_acc`` = 99.26   & ``Params`` = 18,074.
	- Let's try adding a scheduler, "ReduceLROnPlateau". 

### Final.ipynb 
- Target (Changes Made): 
- On adding Regularization with patience = 5, accuracy decreased so increased patience to 10 which improved accuracy. 
- Then added 1 Convolution layer which increased parameters by 900 to increase Receptive Field to 32 as compared to previous 28. 
- Added Image Augmentation technique, transforms.RandomRotation()  . 
- Obsv. 
- Test Accuracy significantly increased and it constantly stays in the range of 99.35 - 99.50. 
- Best ``Train_acc`` = 99.06 and ``Test_acc`` = 99.49   & ``Params`` = 18,994.
- One another convolution layer could be added to further increase receptive field and hence accuracy but it's really not needed. 
- Final objective of using GAP, using less than 20k params, less than 20 epochs and reaching 99.4% test accuracy achieved. 







