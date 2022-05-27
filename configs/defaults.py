# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST
# _C.TASK = CN()
_C = CN()

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C.TASK = ''

_C.GPU = CN()
_C.GPU.USE = True
_C.GPU.ID = [0]

_C.LOG = CN()
_C.LOG.EPOCH_FREQ = 5
_C.LOG.BATCH_FREQ = 5
_C.LOG.DRAW_BOX = True
_C.LOG.DRAW_BOX_FREQ = 50

_C.MODEL = CN()
_C.MODEL.ARCH = ''
_C.MODEL.DETECTOR = CN()
_C.MODEL.DETECTOR.BACKBONE = ''
_C.MODEL.DETECTOR.PRETRAINED_PATH = ''
_C.MODEL.LOAD = ''
_C.MODEL.SAVEDIR = 'results/'

# # -----------------------------------------------------------------------------
# Dataset
# # -----------------------------------------------------------------------------
_C.DATA = CN()
# List of the dataset names for training, as present in paths_catalog.py\
_C.DATA.ROOT = ''
_C.DATA.NAMES = []
_C.DATA.IMAGE_DIRS = []
_C.DATA.GT_DIRS = []
_C.DATA.SYN_NAMES = []
_C.DATA.SYN_TYPES = [[]]
_C.DATA.SYN_IMAGE_DIRS = []
_C.DATA.SYN_GT_DIRS = []
_C.DATA.BATCH_SIZE = 1
_C.DATA.WORKERS = 0
_C.DATA.IMAGE_H = 256
_C.DATA.IMAGE_W = 256
_C.DATA.SCALE_H = 288
_C.DATA.SCALE_W = 288
_C.DATA.AUG = False
_C.DATA.MAX_NUM = 8
_C.DATA.PIN = False

_C.VAL = CN()
_C.VAL.USE = True
_C.VAL.ROOT = ''
_C.VAL.IMAGE_DIRS = []
_C.VAL.GT_DIRS = []

_C.TEST = CN()
_C.TEST.ROOT = ''
_C.TEST.IMAGE_DIRS = []
_C.TEST.GT_DIRS = []
_C.TEST.CHECKPOINT = ''
_C.TEST.SAVE_PATH= "Preds/DetN/"
# # ---------------------------------------------------------------------------- #
# # Solver
# # ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.JOINT = True
_C.SOLVER.MAX_EPOCHS = 300
_C.SOLVER.LR = 0.0001
_C.SOLVER.DECAY_STEPS = 50
_C.SOLVER.LR_DECAY_GAMMA = 0.5
_C.SOLVER.TRANSFORMER_LAYERS = 4
_C.SOLVER.BOX_FEATURE_DIM = 256
_C.SOLVER.CLS_POS_WEIGHT = 1.0
_C.SOLVER.POS_IN_EACH_LAYER = True
_C.SOLVER.ATTENTION_HEADS = 8