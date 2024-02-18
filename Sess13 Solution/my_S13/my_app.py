import torch, torchvision
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.resnet import ResNet18
import gradio as gr
import cv2
from pytorchLightning import LitCIFAR10

#function to get misclassified images 
def gradcam_misclassified(model, device, test_loader, classes, n):
    model.eval()
    misclassified_imgs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            # Check for misclassification
            mis_idxs = (pred != target).nonzero(as_tuple=False).squeeze()
            
            for idx in mis_idxs:
                if len(misclassified_imgs) >= n:
                    break  # Break if we have already collected n misclassified images
                misclassified_imgs.append((data[idx].cpu(), target[idx].item(), pred[idx].item()))

            if len(misclassified_imgs) >= n:
                break  # Break the outer loop if n images are collected

    # Process for Gradio output
    misclassified_processed = []
    for img, actual, predicted in misclassified_imgs:
        # Convert image to displayable format
        img = img.numpy().transpose(1, 2, 0)  # Convert to HWC for visualization
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        label = f"Actual: {classes[actual]}, Predicted: {classes[predicted]}"
        misclassified_processed.append((img, label))
        # misclassified_processed[label] = img

    return misclassified_processed


    

model = ResNet18()
model.load_state_dict(torch.load("PL_saved_model.pth", map_location=torch.device('cpu')), strict=False)

inv_normalize = transforms.Normalize(
mean=[0.4914, 0.4822, 0.4465],
std=[0.2023, 0.1994, 0.2010]
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
def inference(gradcam_yn, n_gradcam, layer_number, transparency, misclassified_yn, n_misclassified, img, top_k): 
        global model 
        # transform = transforms.ToTensor()
        transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((32, 32)),
        ])
        org_img = img
        img = transform(img)
        img = img.unsqueeze(0)
        output = model(img)
        softmax = torch.nn.Softmax(dim=0)
        o = softmax(output.flatten()) #getting top_k values
        top_prob, top_catid = torch.topk(o, top_k)
        confidences = {classes[catid]: prob.item() for prob, catid in zip(top_prob, top_catid)}
        predictions = "\n".join(f"{class_name}: {conf:.2f}" for class_name, conf in confidences.items())

        gradcam_images = [] #contains a list of number of gradcam images
        misclassified_images = []

        if(gradcam_yn): #wants to see gradcam images 
            target_layers = [model.layer2[layer_number]]
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=img, targets=None)
            img = img.squeeze(0)
            img = inv_normalize(img)
            rgb_img = np.transpose(img, (1, 2, 0))
            rgb_img = rgb_img.numpy()
            for i in range(n_gradcam):
                grayscale_cam_i = grayscale_cam[0,:]
                visualization = show_cam_on_image(rgb_img / 255, grayscale_cam_i, use_rgb=True, image_weight = transparency)
                gradcam_images.append(visualization)
        #about misclassified images 
        if(misclassified_yn): 
            model = LitCIFAR10()
            model.setup()
            test_loader = model.test_dataloader()
            misclassified_images_labels = gradcam_misclassified(model, device, test_loader, classes, n_misclassified)
            # print('this is misclassified images labels: ', misclassified_images_labels)
            misclassified_images = [img for img,_ in misclassified_images_labels]
            # labels = [f"Actual: {classes[actual]}, Predicted: {classes[predicted]}" for _, actual, predicted in misclassified_images_labels]
            labels = [label for img, label in misclassified_images_labels]
        # return gradcam_images, [img for img, label in misclassified_images], [label for img, label in misclassified_images], predictions
        return gradcam_images, misclassified_images,labels, predictions


title = "Sess12Solution"
description = "Should fulfill all Sess12 Tasks"
demo = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Radio(["yes", "no"], value="yes", label="GradCAM", info="Do you want to see GradCAM images?"),
        gr.Slider(1, 10, step=1, value=4, label="GradCAM", info="Number of GradCAM images to view:"),
        gr.Slider(-2, -1, value=-2, step=1, label="Layer Selection", info="Select the layer for GradCAM:"),
        gr.Slider(0, 1, step=0.1, value=0.5, label="Opacity", info="Adjust GradCAM opacity:"),
        gr.Radio(["yes", "no"], value="yes", label="Misclassified Images", info="Do you want to see misclassified images?"),
        gr.Slider(1, 10, step=1, value=3, label="Misclassified", info="Number of misclassified images to view:"),
        gr.Image(height=32, width=32, label="Input Image"),
        gr.Slider(1, 10, step=1, value=4, label="Top Predictions", info="Number of top predictions to display:")
    ],
    outputs=[
        gr.Gallery(label="Requested GradCAM Images"),
        gr.Gallery(label="Misclassified Images"),
        gr.Textbox(label="Labels"),
        gr.Textbox(label="Predicted Classes and Confidences")
    ],
    title=title,
    description=description,
)

demo.launch(share = True)
