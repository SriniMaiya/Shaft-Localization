# Training of an Instance Segmentation Model (Mask-RCNN)

    ```python
    def maskrcnn_resnet101_fpn(num_classes=2, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
        backbone  = resnet_fpn_backbone("resnet101", pretrained=pretrained_backbone, trainable_layers=trainable_backbone_layers)

        anchor_generator = AnchorGenerator(sizes=((64,), (76,), (100,), (112,), (128,)),
                                                aspect_ratios=tuple((0.25, 1, 4.0) for _ in range(5))
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
                            mask_roi_pool=mask_roi_pooler,
                            box_detections_per_img=8  )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 4096
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                            hidden_layer,   
                                                            num_classes) 
    ```

Creation of a Mask-RCNN model, by using a ResNet-101 backbone with suitable Mask and bounding box predictors. 
