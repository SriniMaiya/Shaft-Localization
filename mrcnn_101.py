import torch
import utils
import torchvision
from engine import train_one_epoch, evaluate
from transforms import Compose, PILToTensor
from coco_utils import get_coco, get_coco_kp
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def maskrcnn_resnet101_fpn(num_classes=2, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    
    
    backbone  = resnet_fpn_backbone("resnet101", pretrained=pretrained_backbone, trainable_layers=trainable_backbone_layers)
    
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (64,), (76,), (100,)),
                                            aspect_ratios=tuple((0.25, 0.33, 1, 2, 3) for _ in range(5))
                                        )

    # anchor_generator = AnchorGenerator(sizes=((32,), (64,), (76,), (80,), (126,)),
    #                                         aspect_ratios=tuple((0.33, 0.5, 1, 2, 3.33) for _ in range(5))
    #                                     )
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





    # print(model.eval())
    return model
 

def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 2),  
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def main():
    print(torch.cuda.memory_allocated())
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    dataset, _ = get_dataset("coco", "train",
                            transform=Compose([PILToTensor()]),
                                        data_path="datasets/train")
    dataset_test, _ = get_dataset("coco", "val",
                            transform=Compose([PILToTensor()]),
                                        data_path="datasets/val")

    indices_test = torch.randperm(len(dataset_test)).tolist()


    model = maskrcnn_resnet101_fpn(num_classes=2, trainable_backbone_layers=1, pretrained_backbone=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.003,
                                momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=6,
                                                   gamma=0.5)

    num_epochs = 30

    for epoch in range(num_epochs):
        indices = torch.randperm(len(dataset)).tolist()
        dataset_send = torch.utils.data.Subset(dataset, indices[:150])
        dataset_test_send = torch.utils.data.Subset(dataset_test, indices_test[:30])

        
        # print("\t\tIndices:\n",indices,"\n\t\tIndices_test:\n", indices_test)
        data_loader = torch.utils.data.DataLoader(
                            dataset_send, batch_size=1, shuffle=True, num_workers=0,
                            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
                            dataset_test_send, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=utils.collate_fn)
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=30)
        print(torch.cuda.memory_allocated())

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
                
    torch.save(model.state_dict(), f"weights/resnet101_weights_{epoch}.pth")

    
if __name__ == "__main__":
    main()