#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
import numpy as np
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .position_encoding import build_position_encoding
from detectron2.layers import DeformConv

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SparseRepPointsHead(nn.Module):

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.refine = cfg.MODEL.SparseRepPoints.REFINE
        self.delta_xy = cfg.MODEL.SparseRepPoints.DELTA_XY
        self.point_feat_head = PointFeatHead(cfg, input_shape)
        self.predict_head = PredictHead(cfg)

    def forward(self, features):

        if self.delta_xy:
            point_feats, position_feats, center_objectness, topk_xys = self.point_feat_head(features)
            init_xys = torch.cat(topk_xys, dim=1)
        else:
            point_feats, position_feats, center_objectness = self.point_feat_head(features)

        if self.refine:
            point_feats, ref_point_feats = point_feats['init_feat'], point_feats['refine_feat']
            position_feats, ref_position_feats = position_feats['init_feat'], position_feats['refine_feat']

            ref_point_feats = torch.cat(ref_point_feats, dim=2)
            ref_position_feats = torch.cat(ref_position_feats, dim=2)
            ref_class_logits, ref_pred_bboxes = self.predict_head(ref_point_feats, ref_position_feats)

            if self.delta_xy:
                ref_pred_bboxes[:, :, :2] = ref_pred_bboxes[:, :, :2] + init_xys

        point_feats = torch.cat(point_feats, dim=2)
        position_feats = torch.cat(position_feats, dim=2)
        class_logits, pred_bboxes = self.predict_head(point_feats, position_feats)

        if self.delta_xy:
            pred_bboxes[:, :, :2] = pred_bboxes[:, :, :2] + init_xys

        if self.refine:
            return class_logits[None], pred_bboxes[None], ref_class_logits[None], ref_pred_bboxes[
                None], center_objectness
        return class_logits[None], pred_bboxes[None], center_objectness


class ObjectnessHead(nn.Module):

    def __init__(self, channel, k):
        super().__init__()

        self.channel = channel
        self.k = k

        self.conv = nn.Conv2d(self.channel, 1, 3, padding = 1)
        self.norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def get_xy(self, p: torch.Tensor):
        """
        :param p: (N, 1, H, W) or (N, H, W) output of self.forward
        :return xy: (N, k, 2) k xys
        """
        p = p.squeeze(dim = 1)
        N, h, w = p.shape
        p = p.reshape((N, -1))
        values, indices = p.topk(self.k, 1)
        xy = torch.stack((indices // w / float(h), indices % w / float(w)), dim = -1)
        return xy, indices

    def forward(self, feature):
        """
        :param feature: (N, C, H, W) one scale feature of FPN output
        :return p: (N, 1, H, W) probability of object
        """

        feature = self.conv(feature)
        feature = self.norm(feature)
        p = self.sigmoid(feature)

        return p


class PointFeatHead(nn.Module):

    def __init__(self, cfg, input_shape):
        super().__init__()
        d_feat = cfg.MODEL.SparseRepPoints.HIDDEN_DIM
        self.num_points = cfg.MODEL.SparseRepPoints.NUM_POINTS
        self.in_features = cfg.MODEL.SparseRepPoints.IN_FEATURES
        self.top_k = cfg.MODEL.SparseRepPoints.TOP_K
        self.delta_xy = cfg.MODEL.SparseRepPoints.DELTA_XY
        pts_out_dim = 2 * self.num_points

        # Offset Learning.
        self.relu = nn.ReLU(inplace = True)
        self.offset_conv = nn.Conv2d(d_feat, d_feat, 3, 1, 1)
        self.offset_out = nn.Conv2d(d_feat, pts_out_dim, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        # Build Objectness Head.
        self.objectness_heads = nn.ModuleDict(
            {in_feature: ObjectnessHead(input_shape[in_feature].channels, self.top_k) for in_feature in self.in_features})

        # Build Position Embedding.
        self.position_encoding = build_position_encoding(cfg)

        # Refine.
        self.refine = cfg.MODEL.SparseRepPoints.REFINE
        if self.refine:
            # we use deform conv to extract points features
            self.dcn_kernel = int(np.sqrt(self.num_points))
            self.dcn_pad = int((self.dcn_kernel - 1) / 2)
            assert self.dcn_kernel * self.dcn_kernel == self.num_points, \
                'The points number should be a square number.'
            assert self.dcn_kernel % 2 == 1, \
                'The points number should be an odd square number.'
            dcn_base = np.arange(-self.dcn_pad,
                                 self.dcn_pad + 1).astype(np.float64)
            dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
            dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
            dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
                (-1))
            self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

            self.refine_dconv = DeformConv(d_feat, d_feat, self.dcn_kernel, 1, self.dcn_pad)
            # Refine offset learning.
            self.refine_offset_conv = nn.Conv2d(d_feat, d_feat, 3, 1, 1)
            self.refine_offset_out = nn.Conv2d(d_feat, pts_out_dim, 1, 1, 0)

        # Init parameters.
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features):
        batch_size = len(features[0])

        # top_k (x,y)s for each FPN level
        topk_xys = list()
        topk_indices = list()
        center_objectness = list()
        for index, in_feature in enumerate(self.in_features):
            p = self.objectness_heads[in_feature](features[index])  # [batch_size, 1, w, h]
            center_objectness.append(p.squeeze(dim = 1))
            # xy -> [batch_size, top_k, 2], indice -> [batch_size, top_k]
            xy, indice = self.objectness_heads[in_feature].get_xy(p)
            topk_xys.append(xy)
            topk_indices.append(indice)

        # offset learning.
        point_feats = list()
        position_feats = list()
        refine_point_feats = list()
        refine_position_feats = list()
        for i, x in enumerate(features):
            offset = self.sigmoid(self.offset_out(self.relu(self.offset_conv(x))))  # [b, 2*num_points, w, h]
            topk_feat = self.sample_feat(x, offset, topk_xys[i], topk_indices[i], batch_size)
            point_feats.append(topk_feat)
            position_feats.append(self.position_encoding(topk_feat))

            if self.refine:
                dcn_base_offset = self.dcn_base_offset.type_as(x)
                dcn_offset = offset - dcn_base_offset
                x = self.relu(self.refine_dconv(x, dcn_offset))
                refine_offset = self.refine_offset_out(self.relu(self.refine_offset_conv(x)))
                refine_topk_feat = self.sample_feat(x, offset + refine_offset, topk_xys[i], topk_indices[i], batch_size)
                refine_point_feats.append(refine_topk_feat)
                refine_position_feats.append(self.position_encoding(refine_topk_feat))

        if self.refine:
            point_feats = {'init_feat': point_feats, 'refine_feat': refine_point_feats}
            position_feats = {'init_feat': position_feats, 'refine_feat': refine_position_feats}

        if self.delta_xy:
            return point_feats, position_feats, center_objectness, topk_xys

        return point_feats, position_feats, center_objectness

    def sample_feat(self, feature, offset, xy, indice, batch_size):
        topk_idx_repeat = indice[:, :, None].repeat([1, 1, 2 * self.num_points])  # [b, top_k, 2*num_points]
        topk_offset = torch.gather(offset.reshape((batch_size, 2 * self.num_points, -1)).permute(0, 2, 1), 1,
                                   topk_idx_repeat)
        topk_points = topk_offset.reshape((batch_size, -1, self.num_points, 2)) + xy[:, :, None, :].repeat(
            (1, 1, self.num_points, 1))  # [b, top_k, num_points, 2]
        topk_points = 2 * topk_points - 1

        topk_feat = F.grid_sample(feature, topk_points, padding_mode='border')  # [b, C, top_k, num_points], padding
        return topk_feat


class PredictHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        num_classes = cfg.MODEL.SparseRepPoints.NUM_CLASSES  # 80
        num_points = cfg.MODEL.SparseRepPoints.NUM_POINTS
        d_feat = cfg.MODEL.SparseRepPoints.HIDDEN_DIM
        use_focal = cfg.MODEL.SparseRepPoints.USE_FOCAL

        # TODO add
        d_combine = num_points * (d_feat + d_feat)
        self.combine_fcs = nn.ModuleList([
            nn.Linear(d_combine, d_feat, False),
            nn.ReLU(inplace = True),
            nn.Linear(d_feat, d_feat, False),
            nn.ReLU(inplace = True)
        ])

        # Use Focal.
        if use_focal:
            self.class_embed = nn.Linear(d_feat, num_classes)
        else:
            self.class_embed = nn.Linear(d_feat, num_classes + 1)
        self.bbox_embed = MLP(d_feat, d_feat, 4, 3)

        # Init parameters.
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, point_feature, position_feature):
        # point_feature [b, 256, 80, 9]
        b, d_feat, num_object, num_points = point_feature.size()
        d = num_points * d_feat
        # TODO add
        point_feature = point_feature.permute(0, 2, 3, 1).reshape(b, num_object, d)
        position_feature = position_feature.permute(0, 2, 3, 1).reshape(b, num_object, d)

        x = torch.cat([point_feature, position_feature], dim = -1)
        for layer in self.combine_fcs:
            x = layer(x)
        outputs_cls = self.class_embed(x)
        outputs_box = self.bbox_embed(x).sigmoid()

        return outputs_cls, outputs_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
