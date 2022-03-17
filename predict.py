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
    return image


model = maskrcnn_resnet101_fpn(num_classes=2, trainable_backbone_layers=2, pretrained_backbone=True)
weights_pth = "weights/resnet101_weights_GD_IMG2500_OL4096.pth"
model.load_state_dict(torch.load(weights_pth))
model.eval()

# model = instance_segmentation_model(num_classes=2)
# weights_pth = "weights/resnet101_weights_GoogleDrive.pth"

# weights = torch.load(weights_pth)
# model.load_state_dict(weights)
# model.eval()

totensor = transforms.Compose([
                                    transforms.ToTensor(),

                                ])

with torch.no_grad(): 
    images_pth = glob.glob("datasets/real_test/*bmp")
    # images_pth = ["datasets/real_test/image_000116.png"]
    for img_pth in images_pth:
        print(img_pth)
        img = Image.open(img_pth).convert("RGB")
        tensor = normalize(totensor(img))
        tensor = torch.unsqueeze(tensor, 0)
        predictions = model(tensor)
        predictions = predictions[0]
        # print(predictions.keys(), "\n")
        # print(predictions['boxes'])
        # print(predictions['scores'])
        img = np.array(img)

        for i, mask in enumerate(predictions['masks']):
            if predictions['scores'][i] > 0.70:
                mask = mask.mul(255).numpy()
                print(len(mask[0]))
                mask = mask[0]
                ret, thresh = cv2.threshold(mask, 60, 255, 0)
                thresh = thresh.astype(np.uint8)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 1:
                    lengths = [len(c) for c in contours]
                    contours = contours[np.argmax(lengths)]
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                # img = cv2.resize(img, (600,600))
                show = np.hstack([img, mask])
                cv2.imshow("img output", show)
                cv2.waitKey()
                show = None
