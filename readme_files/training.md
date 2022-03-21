# Training of an Instance Segmentation Model (Mask-RCNN)

As PyTorch offers a pretrained MaskRCNN with fpn backbone of ResNet50, a custom model was created by referring [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-your-model). 

Custom anchors sizes and aspect-ratios were defiened to suit the shape of the object, as shown in the code-snippet below. Pretrained ResNet101 backbone was chosen to obtain a faster convergence of loss.

### Mask RCNN model
----
```python3
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def maskrcnn_resnet101_fpn(num_classes=2, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    
    
    backbone  = resnet_fpn_backbone("resnet101", pretrained=pretrained_backbone, trainable_layers=trainable_backbone_layers)
    
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (64,), (76,), (100,)),
                                            aspect_ratios=tuple((0.25, 0.33, 1, 2, 3) for _ in range(5))
                                        )

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
                        mask_roi_pool=mask_roi_pooler, rpn_batch_size_per_image=512 )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 1500
    model.roi_heads.mask_predictor = MaskRCNNPredictor( in_features_mask,
                                                        hidden_layer,   
                                                        num_classes) 

    return model
```

The model is trained for 30 epochs, with 300 iterations each on a dataset of 2500 synthetic images. Learning rate scheduling was applied, decreasing the learning rate by 50% per 6 epochs. 

### Optimizer and Learning rate Scheduler
----
```python
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.003,
                            momentum=0.9, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)
```
### Dataset Loader
----
```python
from coco_utils import get_coco

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 2),  
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

# Create dataloader for loading dataset
dataset, _ = get_dataset("coco", "train",
                        transform=Compose([PILToTensor()]),
                                    data_path="datasets/train")
dataset_test, _ = get_dataset("coco", "val",
                        transform=Compose([PILToTensor()]),
                                    data_path="datasets/val")

# Sample a random of 300 images for train and 60 images for test epoch from the dataloader. 
dataset_send = torch.utils.data.Subset(dataset, indices[:300])
dataset_test_send = torch.utils.data.Subset(dataset_test, indices_test[:60])
```
