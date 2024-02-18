import torch, torchvision
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr
import cv2  
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from my_pl_detection import MyLitYOLOv3 
from utils import * 
import io


IMAGE_SIZE= 416
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_, test_loader, _ = get_loaders(
    train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
)

inv_normalize = transforms.Normalize(
    # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]
    mean= [0,0,0],
    std=[1,1,1]
)
anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(config.DEVICE)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

model = MyLitYOLOv3() 
model.load_state_dict(torch.load("provide path", map_location=torch.device('cpu')), strict =False)
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def my_plot_image(image, boxes): 
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
    # Instead of plt.show(), convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to prevent it from being displayed in the notebook
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    buf.close()

    return img_arr

def inference(input_img, transparency=0.5, thresh = 0.8, iou_thresh = 0.3):
    org_img = input_img 
    input_img = test_transforms(image=input_img)['image'].unsqueeze(0).to(DEVICE)
    x = input_img
    # outputs = model(input_img)
    with torch.no_grad():#from plot couple examples function
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )  
    visualization = my_plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

    #gradcam part(input_img), target layer used would be -1
    target_layers = [model.layer2[-1]]
    cam = GradCAM(model=model,target_layers=target_layers)
    grayscale_cam=cam(input_tensor=input_img, targets=None)
    input_img=input_img.squeeze(0)
    input_img=inv_normalize(input_img)
    rgb_img=np.transpose(input_img, (1,2,0))
    rgb_img=rgb_img.numpy()
    grayscale_cam = grayscale_cam[0, :] 
    visualization2 = show_cam_on_image(org_img/255, grayscale_cam,use_rgb=True,image_weight=transparency)

    #gradcam for 3 sample images
    #for 3 random sample images: 
    sample_images = []
    sample_gradcam_images = []
    for _ in range(3):  # For 3 random sample images
        idx = random.choice(range(len(test_loader.dataset)))
        sample_image, _ = test_loader.dataset[idx]
        sample_image = test_transforms(image=sample_image)['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(sample_image)
            bboxes = [[] for _ in range(sample_image.shape[0])]
            for i in range(3):
                anchor = anchors[i]
                boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S[i], is_preds=True)
                for box in boxes_scale_i[0]:  # Assuming the first (and only) item in batch
                    bboxes[0] += box

        # Move NMS outside the torch.no_grad() block as a logical separation
        nms_boxes = non_max_suppression(bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint")
        
        sample_visualization = my_plot_image(sample_image[0].cpu().permute(1,2,0), nms_boxes)
        sample_images.append(sample_visualization)
        #gradcam part for sample images
        target_layers = [model.layer2[-1]] 
        cam = GradCAM(model = model, target_layers = target_layers) 
        grayscale_cam = cam(input_tensor = sample_image, targets = None)
        grayscale_cam = grayscale_cam[0, :]

        sample_image_transformed = sample_image.squeeze(0)
        sample_image_transformed = inv_normalize(sample_image_transformed)
        rgb_img = np.transpose(sample_image_transformed.cpu().numpy(), (1, 2, 0))
        #apply gradcam visualiation 
        gradcam_visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)
        sample_gradcam_images.append(gradcam_visualization)

    return visualization ,visualization2, sample_images, sample_gradcam_images


#take care of gradio part.
title = "Sess13Solution"
description = "takes user image and gives yolo+gradcam o/p"
demo = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Image(height=32, width=32, label="Input Image"),
        gr.Slider(0, 1, step=0.1, value=0.5, label="Opacity", info="Adjust GradCAM opacity:"),
        gr.Slider(0, 1, step=0.1, value=0.5, label="Threshold", info="Set yolo threshold"),
        gr.Slider(0, 1, step=0.1, value=0.3, label="IOU Threshold", info="Set minimum overlap threshold."),
    ],

    outputs=[
        gr.Image(shape=(32, 32), label="YOLOv3 Output").style(width=128, height=128),
        gr.Image(shape=(32, 32), label="Gradcam Ouput").style(width=128, height=128),
        gr.Gallery(label="Sample images YOLO o/p"),
        gr.Gallery(label="Sample Images gradcam o/p"),
    ],
    title=title,
    description=description,
)

demo.launch(share = True)



