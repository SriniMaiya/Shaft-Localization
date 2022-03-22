from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from mrcnn_101 import maskrcnn_resnet101_fpn
import matplotlib.pyplot as plt
from torchvision import transforms
from copy import deepcopy
import glob, os
from PIL import Image
import torch
import numpy as np




def normalize(image:torch.tensor):
    image = image.float()
    std = torch.std(image)
    mean = torch.mean(image)
    image = torch.divide(torch.subtract(image, mean), std)
    # i_min, i_max = torch.min(image), torch.max(image)
    # image = (image- i_min)/ (i_max - i_min)
    return image

totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions



def predict(img, model):
    cv_img = np.array(img)
    img_tensor = torch.squeeze(torch.tensor(cv_img), 0).permute(2, 0, 1)
    print(img_tensor.shape)
    tensor = normalize(totensor(img))
    tensor = torch.unsqueeze(tensor, 0)
    with torch.no_grad():
        output = model(tensor)
    top_predictions = select_top_predictions(output[0], 0.7)
    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}

    image = draw_bounding_boxes(img_tensor, top_predictions["boxes"], width=1)
    masks = top_predictions["masks"].ge(0.5).squeeze(1)
    print(masks.shape)
    image = draw_segmentation_masks(image,masks, alpha=0.6)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    model = maskrcnn_resnet101_fpn(num_classes=2, trainable_backbone_layers=5, pretrained_backbone=True)
    weights_pth = "weights/resnet101_weights_10.pth"
    model.load_state_dict(torch.load(weights_pth))
    model.eval()

    global pth, img_name 

    images_pth = glob.glob("datasets/real_test/*bmp")
    if not os.path.isdir("datasets/real_test/Results"):
        os.makedirs("datasets/real_test/Results", exist_ok=True)
    for img_pth in images_pth:
        img_name = img_pth.split("/")[-1]
        img = Image.open(img_pth).convert("RGB")
        predict(img, model)

