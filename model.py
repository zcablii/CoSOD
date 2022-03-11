# import some common detectron2 utilities
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.structures import Boxes
from detectron2.config import get_cfg
from detectron2 import model_zoo

import matplotlib.pyplot as plt
import numpy as np
# from skimage.transform import resize

# import some common libraries
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch

from einops import rearrange
import math

from torchvision.ops import nms

from utils.util import filter_boxes_by_prob, write_boxes_imgs

setup_logger()


# mask_rcnn_R_101_C4_3x.yaml
# DetectionCheckpointer(model).load("R_101_C4.pkl")

class RPNet(nn.Module):
    def __init__(self, mode='train', backbone='R_101_C4', backbone_path='', draw_box=False, output_dim = 256):
        super(RPNet, self).__init__()
        self.mode = mode
        self.draw_box = draw_box
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_" + backbone + "_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_" + backbone + "_3x.yaml")

        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(backbone_path + backbone + ".pkl")
        self.model.eval()
        # del self.model.roi_heads.box_predictor
        del self.model.roi_heads.mask_head
        self.output_dim = output_dim
        self.feat_conv = nn.Conv2d(2048, self.output_dim, kernel_size=7, stride=1, padding=0, bias=False)
        self.box_feature_layer = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(self.output_dim),
            nn.Dropout(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim)
            )

        self.class_feature_layer = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(81),
            nn.Dropout(),
            nn.Linear(81, self.output_dim),
            nn.BatchNorm1d(self.output_dim)
            )

    def forward(self, image_Input):

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        activation = {}
        self.model.backbone.res2.register_forward_hook(get_activation('res2'))  # torch.Size([8, 256, 64, 64])
        self.model.backbone.res3.register_forward_hook(get_activation('res3'))  # torch.Size([8, 512, 32, 32])
        self.model.backbone.res4.register_forward_hook(get_activation('res4'))  # torch.Size([8, 1024, 16, 16])

        self.model.eval()
        with torch.no_grad():
            # don't forget to preprocess
            images = self.model.preprocess_image(image_Input)
            # cnn features res4 features['res4']: [n,1024,h,w]
            features = self.model.backbone(images.tensor)
            # RPN  proposals[0].proposal_boxes are Boxes
            proposals, _ = self.model.proposal_generator(images, features, None)
            features_ = [features[f] for f in self.model.roi_heads.in_features]  # [tensor [n,1024,h,w]]
            # print(features_[0].size())

            # Apply NMS on proposed boxes, select 3 to 30 boxes
            proposals_objectness_logits = filter_boxes_by_prob([x.objectness_logits for x in proposals])
            selected_box_num = [len(i) for i in proposals_objectness_logits]
            proposals_boxes = [x.proposal_boxes[:i] for x, i in zip(proposals, selected_box_num)]
            nms_boxes_inds = [nms(proposals_box.tensor, proposals_objectness_logit, 0.3) for
                              (proposals_box, proposals_objectness_logit) in
                              zip(proposals_boxes, proposals_objectness_logits)]

            nms_boxes_ = [[proposals_boxes[i].tensor[index] for index in nms_boxes_ind] for i, nms_boxes_ind in
                          enumerate(nms_boxes_inds)]
            # print(nms_boxes_)
            nms_boxes = [Boxes(torch.cat(nms_boxes_[i]).reshape(-1, 4)) for i in range(len(nms_boxes_))]
            # keep this list for box-image relation
            eachimg_selected_box_nums = [len(boxes) for boxes in nms_boxes]

            # [n,1024,14,14] pooler.py line ~220
            box_features_ = self.model.roi_heads.pooler(features_, nms_boxes)
        if self.mode == 'train':
            self.model.train()
        
        box_features = self.model.roi_heads.res5(box_features_)# features of all 1k candidates [n*1000, 2048,7,7]
        ## box_features = box_features.mean(dim=[2, 3])# [n*1000, 2048] need to mean this value
        # box_features = self.feat_conv(box_features).squeeze(dim=3).squeeze(dim=2)
        # box_features = self.box_feature_layer(box_features)
        
        predictions = self.model.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))[0]
        box_features = self.class_feature_layer(predictions)

        return nms_boxes, box_features, eachimg_selected_box_nums, activation


class PosEmbedding(nn.Module):
    def __init__(self, max_num, box_feat_dim):
        super().__init__()
        positions = torch.rand(max_num, box_feat_dim).cuda()
        # for ind, i in enumerate (positions):
        #     positions[ind][ind] = 1
        self.positions = nn.Parameter(positions)

    def forward(self, eachimg_selected_box_nums, box_features: Tensor) -> Tensor:
        inds = 0
        for ind, i in enumerate(eachimg_selected_box_nums):
            box_features[inds:i + inds] = box_features[inds:i + inds]/i + self.positions[ind]
            inds += i
        return box_features


# towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class ResidualSub(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x -= res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                # nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoderFirstBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualSub(nn.Sequential(
                # nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        # super().__init__(*([TransformerEncoderFirstBlock(**kwargs)]+[TransformerEncoderBlock(**kwargs) for _ in range(depth-1)]))   #1
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])  #0


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers

        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, layers):
        # Bottom-up
        c2 = layers['res2']  # torch.Size([8, 256, 64, 64])
        c3 = layers['res3']  # torch.Size([8, 512, 32, 32])
        c4 = layers['res4']  # torch.Size([8, 1024, 16, 16])

        # Top-down
        p4 = self.toplayer(c4)  # torch.Size([8, 256, 16, 16])
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4


class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        feature_channels = 64
        self.bot_layer = nn.Conv2d(k, feature_channels, 1, 1, 0)
        self.score = nn.Conv2d(feature_channels + 3, 1, 3, 1, 1)
        self.smooth = nn.Conv2d(feature_channels + 3, feature_channels + 3, kernel_size=5, stride=1, padding=2)

    def _upsample_cat(self, x, y):
        _, _, H, W = y.size()
        return torch.cat((F.upsample(x, size=(H, W), mode='bilinear'), y), dim=1)

    def forward(self, x, rgb, x_size=None):
        x = self.bot_layer(x)
        _, _, H, W = rgb.size()
        rgb = F.upsample(rgb, size=(H // 2, W // 2), mode='bilinear')
        x = self.smooth(self._upsample_cat(x, rgb))
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size, mode='bilinear', align_corners=True)
        return x


class MultiHeadCrossSimilarity(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qemb = nn.Linear(emb_size, emb_size)
        self.kemb = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.sigmod = nn.Sigmoid()

    def forward(self, eachimg_selected_box_nums, x: Tensor) -> Tensor:
        q = self.qemb(x)[0] # shape 1,k,d
        k = self.kemb(x)[0]
        max_ = q.size(0) - 1
        box_id = 0
        matrs = []
        querys = []
        keys = []
        for boxNum in eachimg_selected_box_nums:
            this_qs = q[box_id:box_id+boxNum]
            this_qs = rearrange(this_qs, "b (h d) -> b h d", h=2)
            querys.append(this_qs)
            this_ks = k[box_id:box_id+boxNum]
            this_ks = rearrange(this_ks, "b (h d) -> b h d", h=2)
            keys.append(this_ks)
            box_id+=boxNum

        for i, qs in enumerate(querys):
            similarity_l = []
            for j, ks in enumerate(keys):
                if i==j:
                    continue
                similarity = torch.einsum('qhd, khd -> qhk', qs, ks)
            
                similarity_mean = similarity.mean(-1) 
                similarity_max = similarity.max(-1)[0]
                similarity_min = similarity.min(-1)[0]
                similarity = torch.stack([similarity_mean,similarity_max,similarity_min])
                
                similarity_l.append(similarity) 
           
            similarity_l = torch.stack(similarity_l) #imgk * 3 * objq * h
          
            
            avg_sim_matr = similarity_l.mean(0) # 3 * objq * h
            max_sim_matr = similarity_l.max(0)[0]
            min_sim_matr = similarity_l.min(0)[0]
            matr = torch.stack([avg_sim_matr,max_sim_matr,min_sim_matr]) 
            matr = rearrange(matr, 'n m o h -> o (n m h)', h = 2, n=3,m=3) # objq * (3*3*h)
            matrs.append(matr) # 

        matrs = torch.cat(matrs) # all objs * (3*3*h)

        return matrs

        
class CoS_Det_Net(nn.Module):
    def __init__(self, cfg, draw_box=False):
        super(CoS_Det_Net, self).__init__()
        self.cfg = cfg
        self.box_feat_dim = self.cfg.SOLVER.BOX_FEATURE_DIM
        self.det_net = RPNet(mode='train',
                             backbone=cfg.MODEL.DETECTOR.BACKBONE,
                             backbone_path=cfg.MODEL.DETECTOR.PRETRAINED_PATH,
                             draw_box=draw_box,
                             output_dim= self.box_feat_dim )
        self.pos_e = PosEmbedding(cfg.DATA.MAX_NUM, self.box_feat_dim )
        # self.trans_encoder = TransformerEncoder(depth=self.cfg.SOLVER.TRANSFORMER_LAYERS, emb_size=self.box_feat_dim )  # 0
        self.xatt = MultiHeadCrossSimilarity(self.box_feat_dim)
        self.fpn = FPN()
        self.score_Layer = ScoreLayer(256)
        # self.squeeze_features = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048, self.box_feat_dim),
        #     nn.BatchNorm1d(self.box_feat_dim)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.box_feat_dim, self.box_feat_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.box_feat_dim),
            nn.Dropout(),
            nn.Linear(self.box_feat_dim, self.box_feat_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.box_feat_dim),
            nn.Dropout(),
            nn.Linear(self.box_feat_dim, 1)
        )

        self.xatt_classifier = nn.Sequential(
            nn.Linear(18, 18),
            nn.ReLU(True),
            nn.BatchNorm1d(18),
            nn.Dropout(),
            nn.Linear(18, 1)
        )
        self.layerNorm = nn.LayerNorm(self.box_feat_dim)

    def forward(self, images):
        nms_boxes, box_features, eachimg_selected_box_nums, activation = self.det_net(images)
        # box_features = self.squeeze_features(box_features)
        att_features = self.pos_e(eachimg_selected_box_nums, box_features) # 0
        if self.cfg.SOLVER.POS_IN_EACH_LAYER:
            for each_layer in self.trans_encoder: #1
                att_features = self.layerNorm(att_features)#1
                att_features = self.pos_e(eachimg_selected_box_nums, att_features)#1
                att_features = each_layer(att_features.reshape(1, -1,  self.box_feat_dim)).reshape(-1,  self.box_feat_dim)#1
        else:
            att_features = self.xatt(eachimg_selected_box_nums,att_features.reshape(1, -1, self.box_feat_dim))
        pred_vector = self.xatt_classifier(att_features)
        if math.isnan(pred_vector[0][0]):
            print("box_features: ", box_features)
            print("att_features: ", att_features)
            print("nan, value exploded")
            assert False
            # "if appears nan, try to set a smaller lr"
        (p2, p3, p4) = self.fpn(activation)
        images = torch.stack([image['image'] for image in images])

        bmap = self.score_Layer(p2, images, x_size=[256, 256])

        return nms_boxes, pred_vector, bmap

