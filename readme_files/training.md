# Training of an Instance Segmentation Model (Mask-RCNN)

As PyTorch offers a pretrained MaskRCNN with fpn backbone of ResNet50, a custom model was created by referring [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-your-model). 

Custom anchors sizes and aspect-ratios were defiened to suit the shape of the object, as shown in the code-snippet below. Pretrained ResNet101 backbone was chosen to obtain a faster convergence of loss.

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

The model was trained for 30 epochs, with 300 iterations each on a dataset of 2500 synthetic images. Learning rate scheduling was applied, decreasing the learning rate by 50% per 6 epochs. 

```python
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.003,
                            momentum=0.9, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)
```

