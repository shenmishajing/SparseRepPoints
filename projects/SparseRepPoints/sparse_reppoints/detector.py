#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import SparseRepPointsHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

__all__ = ["SparseRepPoints"]


@META_ARCH_REGISTRY.register()
class SparseRepPoints(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.SparseRepPoints.IN_FEATURES  # ["p2", "p3", "p4", "p5"]
        self.num_classes = cfg.MODEL.SparseRepPoints.NUM_CLASSES  # 80
        self.k = cfg.MODEL.SparseRepPoints.TOP_K  # 20
        self.num_objects = len(self.in_features) * self.k

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Sparse RepPoints Head.
        self.sparse_head = SparseRepPointsHead(cfg=cfg, input_shape=self.backbone.output_shape())

        # Loss parameters:
        objectness_weight = cfg.MODEL.SparseRepPoints.OBJECTNESS_WEIGHT
        class_weight = cfg.MODEL.SparseRepPoints.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRepPoints.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRepPoints.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRepPoints.NO_OBJECT_WEIGHT
        self.use_focal = cfg.MODEL.SparseRepPoints.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, "loss_objectness": objectness_weight}

        losses = ["labels", "boxes", "objectness"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prediction.
        outputs_class, outputs_coord, center_objectness = self.sparse_head(features)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'center_objectness': center_objectness}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, [p.shape[1:] for p in output['center_objectness']])

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets, objectness_shape):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            if len(targets_per_image.gt_boxes.tensor) > 0:
                target["objectness"] = []
                for h, w in objectness_shape:
                    xy = \
                    torch.stack(((torch.arange(h, dtype=torch.float, device=self.device) / h)[:, None].expand(-1, w),
                                 (torch.arange(w, dtype=torch.float, device=self.device) / w)[None, :].expand(h, -1)),
                                dim=-1)[
                        None]
                    boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
                    l = xy[..., 1] - boxes[:, None, None, 0]
                    r = boxes[:, None, None, 2] - xy[..., 1]
                    t = xy[..., 0] - boxes[:, None, None, 1]
                    b = boxes[:, None, None, 3] - xy[..., 0]
                    in_box = (l > 0) & (r > 0) & (t > 0) & (b > 0)
                    obj = torch.zeros((boxes.shape[0], h, w), dtype=torch.float, device=self.device)
                    obj[in_box] = torch.sqrt(torch.min(l, r) / torch.max(l, r) * torch.min(t, b) / torch.max(t, b))[
                        in_box]
                    objectness, _ = torch.max(obj, dim=0)
                    target["objectness"].append(objectness)
            else:
                target["objectness"] = [torch.zeros((h, w), dtype=torch.float, device=self.device) for h, w in
                                        objectness_shape]
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_objects, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_objects, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])

                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
