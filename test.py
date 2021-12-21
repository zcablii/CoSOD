import torch

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from torchvision.ops import nms
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes
from util import *

# mask_rcnn_R_101_C4_3x.yaml
# DetectionCheckpointer(model).load("R_101_C4.pkl")

im1 = cv2.imread('./imgs/4.jpg')
image = im1
height, width = image.shape[:2]
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

# print(im1.shape,np.array(image.permute(1, 2, 0)).shape) #(426, 640, 3) (426, 640, 3)

inputs = [{"image": image, "height": height, "width": width}]
im2 = cv2.imread('./imgs/5.jpg')
image = im2
height, width = image.shape[:2]
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
inputs.append({"image": image, "height": height, "width": width})
im3 = cv2.imread('./imgs/1.jpg')
image = im3
height, width = image.shape[:2]
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
inputs.append({"image": image, "height": height, "width": width})


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")

model = build_model(cfg)

DetectionCheckpointer(model).load("./models/R_101_C4.pkl")
model.eval()

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}
model.backbone.stem.register_forward_hook(get_activation('stem')) 
model.backbone.res2.register_forward_hook(get_activation('res2')) 
model.backbone.res3.register_forward_hook(get_activation('res3')) 
model.backbone.res4.register_forward_hook(get_activation('res4')) 

with torch.no_grad():
    images = model.preprocess_image(inputs)  # don't forget to preprocess
    features = model.backbone(images.tensor)  # cnn features res4 features['res4']: [n,1024,h,w]
    proposals, _ = model.proposal_generator(images, features, None)  # RPN  proposals[0].proposal_boxes are Boxes
    features_ = [features[f] for f in model.roi_heads.in_features] #[tensor [n,1024,h,w]]

    # Apply NMS on proposed boxes, select 3 to 30 boxes
    proposals_objectness_logits = filter_boxes_by_prob([x.objectness_logits for x in proposals])
    selected_box_num = [len(i) for i in proposals_objectness_logits]
    proposals_boxes = [x.proposal_boxes[:i] for x,i in zip(proposals,selected_box_num)]

    nms_boxes_inds = [nms(proposals_box.tensor, proposals_objectness_logit, 0.3) for (proposals_box,proposals_objectness_logit) in zip(proposals_boxes,proposals_objectness_logits)]


    nms_boxes_ = [[proposals_boxes[i].tensor[index] for index in nms_boxes_ind] for i,nms_boxes_ind in enumerate(nms_boxes_inds) ]
    nms_boxes = [Boxes(torch.cat(nms_boxes_[i]).reshape(-1,4)) for i in range(len(nms_boxes_))] 
    eachimg_selected_box_nums = [len(boxes) for boxes in nms_boxes]# keep this list for box-image relation 
    print(eachimg_selected_box_nums)
    write_boxes_imgs(nms_boxes,inputs)

    box_features_ = model.roi_heads.pooler(features_, nms_boxes) #  [n*1000,1024,14,14] pooler.py line ~220
    
    box_features = model.roi_heads.res5(box_features_)  # features of all 1k candidates [n*1000, 2048,7,7]
    box_features = box_features.mean(dim=[2, 3]) #####![n*1000, 2048] need to mean this value
    print(box_features.shape)
    


    # box_features_ = []
    # for (img_boxes,feature_map) in zip(nms_boxes,features['res4']):
    #     pairwise_ioa_mat = Boxes.pairwise_ioa(img_boxes,img_boxes)
    #     order = pairwise_ioa_mat.sum(dim=0).argsort(descending=True)
    #     each_img_boxes_features = roi_cut(feature_map,img_boxes,order) # implement roi cut!
    #     box_features_.append(each_img_boxes_features)

    # # need to reshape box_features_ to [n,1024,14,14]
    # box_features = model.roi_heads.res5(box_features_)  # features of all 1k candidates [n*1000, 2048,7,7]
    # box_features = box_features.mean(dim=[2, 3]) #####![n*1000, 2048] need to mean this value
# print(nms_boxes)