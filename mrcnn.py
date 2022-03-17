import torch
import torchvision
import torchvision.transforms as T
from transforms import Compose, PILToTensor
from coco_utils import get_coco, get_coco_kp
import torchvision.datasets.coco as coco
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import cv2
import argparse
from PIL import Image


def instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,   
                                                        num_classes)
    
    print(model.eval())
    return model


train_img_dir = "dataset_COCODetection/train/images/"
train_annot_dir = "dataset_COCODetection/train/annotations.json" 


val_img_dir = "dataset_COCODetection/val/images/"
val_annot_dir = "dataset_COCODetection/val/annotations.json"

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    
    dataset, _ = get_dataset("coco", "train",transform=Compose([PILToTensor()]), data_path="datasets/train")
    dataset_test, _ = get_dataset("coco", "val",transform=Compose([PILToTensor()]), data_path="datasets/val")

    indices_test = torch.randperm(len(dataset_test)).tolist()


    # get the model using our helper function
    model = instance_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=4,
                                                   gamma=0.5)

    num_epochs = 15

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        indices = torch.randperm(len(dataset)).tolist()
        dataset_send = torch.utils.data.Subset(dataset, indices[:250])
        dataset_test_send = torch.utils.data.Subset(dataset_test, indices_test[:-50])

        
        # print("\t\tIndices:\n",indices,"\n\t\tIndices_test:\n", indices_test)
        data_loader = torch.utils.data.DataLoader(
                            dataset_send, batch_size=1, shuffle=True, num_workers=1,
                            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
                            dataset_test_send, batch_size=1, shuffle=False, num_workers=1,
                            collate_fn=utils.collate_fn)

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "weights/weights.pth")
if __name__ == "__main__":
    main()