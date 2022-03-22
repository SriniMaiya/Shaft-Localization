from copy import deepcopy
from math import atan2, sqrt, cos, sin, pi
import shutil
from torchvision.utils import draw_segmentation_masks
import os
from mrcnn_101 import maskrcnn_resnet101_fpn
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import glob
import cv2

def normalize(image:torch.tensor):
    image = image.float()
    std = torch.std(image)
    mean = torch.mean(image)
    image = torch.divide(torch.subtract(image, mean), std)
    # i_min, i_max = torch.min(image), torch.max(image)
    # image = (image- i_min)/ (i_max - i_min)
    return image


model = maskrcnn_resnet101_fpn(num_classes=2, trainable_backbone_layers=5, pretrained_backbone=True)
weights_pth = "weights_/resnet101_weights_24.pth"
model.load_state_dict(torch.load(weights_pth))
model.eval()

# model = instance_segmentation_model(num_classes=2)
# weights_pth = "weights/resnet101_weights_GoogleDrive.pth"

# weights = torch.load(weights_pth)
# model.load_state_dict(weights)
# model.eval()

def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions["labels"]
    boxes = predictions['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

CATEGORIES = """BACKGROUND,shaft""".split(",")
def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions['boxes']

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2].numpy().astype(np.uint8)
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.25, (100, 0, 0), 1, cv2.LINE_AA
        )

    return image, scores


def overlay_mask(image, predictions, orientation_img):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions["masks"].ge(0.5).mul(255).byte().numpy()
    labels = predictions["labels"]
    confidence = predictions["scores"].tolist()
    colors = compute_colors_for_labels(labels).tolist()
    txt_name = img_name.split(".")[0]+".txt"

    if os.path.isfile(pth+f"/{txt_name}"):
        os.remove(pth+f"/{txt_name}") 
    
    for mask, color, conf in zip(masks, colors, confidence):

        _, thresh = cv2.threshold(mask, 60, 255, 0)
        thresh = thresh[0].astype(np.uint8)
        contours, _ = cv2.findContours(
                                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        if len(contours) > 1:
            lengths = [len(c) for c in contours ]
            contours = [contours[np.argmax(lengths)]]
        
        orientation_img, angle, cntr = getOrientations(contours[0], orientation_img)
        image = cv2.drawContours(image, contours, -1, color, 2)
            
        with open(pth+"/Results"+f"/{txt_name}", "a") as f:
            f.write(f"0 {round(cntr[0], 4)} { round(cntr[1], 4)} {round(-angle, 4)} {round(conf, 4)} \n") 

    cv2.imwrite(pth+"/Results"+f"/{img_name}", orientation_img)
      
    composite = image

    return composite, orientation_img

totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
def predict(img, model):
    cv_img = np.array(img)
    cv_orig = np.array(deepcopy(img))
    tensor = normalize(totensor(img))
    tensor = torch.unsqueeze(tensor, 0)
    with torch.no_grad():
        output = model(tensor)
    top_predictions = select_top_predictions(output[0], 0.9)
    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
    result = deepcopy(cv_img)
    result = overlay_boxes(result, top_predictions)
    if 'masks' in top_predictions:
        result, orientation_img  = overlay_mask(result, top_predictions, cv_orig)

    result, scores = overlay_class_names(result, top_predictions)

    return result, output, top_predictions, orientation_img

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1]-q[1], p[0]- q[0])
    hypotenuse = sqrt((p[1]- q[1])**2 + (p[0]- q[0])**2 )
    q[0] = p[0] - scale*hypotenuse*cos(angle)
    q[1] = p[1] - scale*hypotenuse*sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

def getOrientations(pts, img):
    sz = len(pts)
    data = np.empty((sz, 2), dtype=np.int16)
    for i in range(sz):
        data[i, 0] = pts[i, 0, 0]
        data[i, 1] = pts[i, 0, 1]
    mean, eigVec, eigVal = cv2.PCACompute2(data=data, mean = None)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    cv2.drawMarker(img, cntr,(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1 )
    p1 = (cntr[0] + 0.02 * eigVec[0,0] * eigVal[0,0], cntr[1] + 0.02 * eigVec[0,1] * eigVal[0,0])
    p2 = (cntr[0] - 0.02 * eigVec[1,0] * eigVal[1,0], cntr[1] - 0.02 * eigVec[1,1] * eigVal[1,0])
    drawAxis(img, cntr, p1, (0,0,255), 1)
    angle = atan2(eigVec[0,1], eigVec[0,0]) # orientation in radians
    angle = angle * 180 / pi
    return img, angle, cntr





if __name__ == "__main__":
    global pth, img_name 
    pth = "readme_files/synthetic_test"

    images_pth = glob.glob(pth+"/*png")
    if not os.path.isdir(pth+"/Results"):
        os.makedirs(pth+"/Results", exist_ok=True)
    for img_pth in images_pth:
        img_name = img_pth.split("/")[-1]
        img = Image.open(img_pth).convert("RGB")
        result, output, top_pred, orientations = predict(img, model)
        result = cv2.resize(result, (600, 600), interpolation=cv2.INTER_LANCZOS4)
        orientations = cv2.resize(orientations, (600, 600), interpolation=cv2.INTER_LANCZOS4)
        disp = np.hstack([result, orientations])
        cv2.imshow("Output", disp)
        disp_pth = pth+"/Results/"+img_name.split(".")[0]+"_pred."+img_name.split(".")[-1]
        print(disp_pth)
        cv2.imwrite(disp_pth, disp)
        cv2.waitKey(500)