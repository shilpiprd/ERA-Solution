### Task 
- Make a model with <= 8000 parameters and achieve 99.4% accuracy with <=15 epochs 

### Junior Skeleton 
- Target( Steps Taken): 
- Initially I tried a set of conv. layers but final RF achieved was 30 . 
- Then I added padding in one layer to increase one conv layer after 2nd max pooling, which increased RF to further 38. 
- Obsv: 
- With 15 epochs and No ReLU, Best ``Train_Acc``: 98.88 and ``Test_Acc``: 98.67 & ``Params``: 6,008. 
- After adding ReLU,  Best ``Train_Acc``: 99.24 and ``Test_Acc``: 98.86 & ``Params``: 6,008. 
- We notice that training and test accuracy improved greatly. 
- Also there is a significant overfitting with which we'd deal later. 
- Next I'll try adding BatchNorm and Dropout. 

