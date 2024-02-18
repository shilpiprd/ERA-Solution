import torch, torchvision
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from resnet import ResNet18
import gradio as gr

model = ResNet18()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')), strict=False)

inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def inference(input_img, transparency = 0.5, target_layer_number = -1):
    transform = transforms.ToTensor()
    org_img = input_img
    input_img = transform(input_img)
    input_img = input_img
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(outputs.flatten())
    confidences = {classes[i]: float(o[i]) for i in range(10)}
    _, prediction = torch.max(outputs, 1)
    target_layers = [model.layer2[target_layer_number]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    img = input_img.squeeze(0)
    img = inv_normalize(img)
    rgb_img = np.transpose(img, (1, 2, 0))
    rgb_img = rgb_img.numpy()
    visualization = show_cam_on_image(org_img/255, grayscale_cam, use_rgb=True, image_weight=transparency)
    return confidences, visualization

title = "CIFAR10 trained on ResNet18 Model with GradCAM"
description = "A simple Gradio interface to infer on ResNet model, and get GradCAM results"
examples = [["cat.jpg", 0.5, -1], ["dog.jpg", 0.5, -1]]
demo = gr.Interface(
    inference, 
    inputs = [gr.Image(shape=(32, 32), label="Input Image"), gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM"), gr.Slider(-2, -1, value = -2, step=1, label="Which Layer?")], 
    outputs = [gr.Label(num_top_classes=3), gr.Image(shape=(32, 32), label="Output").style(width=128, height=128)],
    title = title,
    description = description,
    examples = examples,
)
demo.launch()