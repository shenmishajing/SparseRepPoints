# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_sparsereppoints_config(cfg):
    """
    Add config for SparseRepPoints.
    """
    cfg.MODEL.SparseRepPoints = CN()
    cfg.MODEL.SparseRepPoints.NUM_CLASSES = 80
    cfg.MODEL.SparseRepPoints.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.SparseRepPoints.HIDDEN_DIM = 256

    # Objectness Head.
    cfg.MODEL.SparseRepPoints.OBJECTNESS_SHARE = False
    cfg.MODEL.SparseRepPoints.TOP_K = 20

    # Position Encoding.
    cfg.MODEL.SparseRepPoints.POSITION_EMBEDDING = "learned"

    # Offset Learning.
    cfg.MODEL.SparseRepPoints.NUM_POINTS = 9
    cfg.MODEL.SparseRepPoints.REFINE = True
    cfg.MODEL.SparseRepPoints.GRADIENT_MUL = 0.1

    # Predict delta x, delta y, w, h
    cfg.MODEL.SparseRepPoints.DELTA_XY = False

    # Loss.
    cfg.MODEL.SparseRepPoints.OBJECTNESS_WEIGHT = 2.0
    cfg.MODEL.SparseRepPoints.CLASS_WEIGHT = 2.0
    cfg.MODEL.SparseRepPoints.GIOU_WEIGHT = 2.0
    cfg.MODEL.SparseRepPoints.L1_WEIGHT = 5.0
    cfg.MODEL.SparseRepPoints.DEEP_SUPERVISION = True
    cfg.MODEL.SparseRepPoints.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.SparseRepPoints.USE_FOCAL = True
    cfg.MODEL.SparseRepPoints.ALPHA = 0.25
    cfg.MODEL.SparseRepPoints.GAMMA = 2.0
    cfg.MODEL.SparseRepPoints.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

