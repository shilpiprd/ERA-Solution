# Task 
1. Move your S10 assignment to Lightning first and then to Spaces such that:
(You have retrained your model on Lightning)
You are using Gradio
Your spaces app has these features:
ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
ask whether he/she wants to view misclassified images, and how many
allow users to upload new images, as well as provide 10 example images
ask how many top classes are to be shown (make sure the user cannot enter more than 10)
Add the full details on what your App is doing to Spaces README 

### Explanation: 
1. I converted the Sess10 code into Pytorch Lightning. 
2. Made gradio such that for gradcam, users have the option of choosing how many gradcam images they wanna see of the image they're uploading. This doesn't make sense at the moemnt, since multiple gradcam images of the same uploaded image doesn't give any additional information. Furthermore, they have the opportunity to choose opactiy value and layer value of the model from which gradcam is required. 
3. For misclassified images, they've the right to choose how many misclassified images they wanna see. This depends on the weight from which the model has been loaded. 
4. example images has not been implemented in this solution. 
5. the users have the right to choose how many top predicted classes they wanna see of the image they uploaded. 
