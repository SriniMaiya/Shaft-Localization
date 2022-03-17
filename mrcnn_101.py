import torch
import utils
import torchvision
from torchvision.transforms import Normalize
from torch import nn
from engine import train_one_epoch, evaluate
from transforms import Compose, PILToTensor
from coco_utils import get_coco, get_coco_kp
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


torch.autograd.set_detect_anomaly(True) 

def maskrcnn_resnet101_fpn(num_classes=2, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    
    
    backbone  = resnet_fpn_backbone("resnet101", pretrained=pretrained_backbone, trainable_layers=trainable_backbone_layers)
    ##Original
    anchor_generator = AnchorGenerator(sizes=((64,), (76,), (100,), (112,), (128,)),
                                            aspect_ratios=tuple((0.25, 1, 4.0) for _ in range(5))
                                        )
    ##Mod1
    # anchor_generator = AnchorGenerator(sizes=((80,), (90,), (100,), (110,), (120,)),
    #                                         aspect_ratios=tuple((0.33, 1.0, 1.5,2,3) for _ in range(5))
    ##Mod2
    # anchor_generator = AnchorGenerator(sizes=((80,), (90,), (100,), (110,), (120,)),
    #                                         aspect_ratios=tuple((0.33, 0.4, 1, 2.5, 3,) for _ in range(5)))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=7,
                                                         sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                              output_size=14,
                                                              sampling_ratio=2)
                                                                 

    model = MaskRCNN(backbone=backbone, num_classes=num_classes, 
                        min_size=256, max_size=256,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool= roi_pooler,
                        mask_roi_pool=mask_roi_pooler, )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 4096
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,   
                                                        num_classes) 
    


    # print(model.eval())
    return model

model = maskrcnn_resnet101_fpn(num_classes=2, trainable_backbone_layers=5, pretrained_backbone=True)


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

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    # dataset,target = coco.CocoDetection(train_img_dir, train_annot_dir, transform=get_transform(train=True))
    # dataset_test, target = coco.CocoDetection(val_img_dir, val_annot_dir, transform=get_transform(train=False))
    dataset, _ = get_dataset("coco", "train",
                            transform=Compose([PILToTensor()]),
                                        data_path="datasets/train")
    dataset_test, _ = get_dataset("coco", "val",
                            transform=Compose([PILToTensor()]),
                                        data_path="datasets/val")

    
    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # indices = torch.randperm(len(dataset_test)).tolist()
    # print(indices, len(indices))

    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders


    # get the model using our helper function

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0001)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)

    num_epochs = 15

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        indices = torch.randperm(len(dataset)).tolist()
        dataset_send = torch.utils.data.Subset(dataset, indices[:200])
        # dataset_test_send = torch.utils.data.Subset(dataset_test, indices_test[:-50])

        
        # print("\t\tIndices:\n",indices,"\n\t\tIndices_test:\n", indices_test)
        data_loader = torch.utils.data.DataLoader(
                            dataset_send, batch_size=2, shuffle=True, num_workers=1,
                            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
                            dataset_test, batch_size=2, shuffle=False, num_workers=1,
                            collate_fn=utils.collate_fn)

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "weights/resnet101_weights.pth")
if __name__ == "__main__":
    main()