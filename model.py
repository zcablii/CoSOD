# import some common detectron2 utilities
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.structures import Boxes
from detectron2.config import get_cfg
from detectron2 import model_zoo

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

# import some common libraries
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch

from einops import rearrange
import math

from torchvision.ops import nms

from CoSOD.utils.util import filter_boxes_by_prob, write_boxes_imgs

setup_logger()


# mask_rcnn_R_101_C4_3x.yaml
# DetectionCheckpointer(model).load("R_101_C4.pkl")

class RPNet(nn.Module):
    def __init__(self, mode='train', backbone='R_101_C4', backbone_path='', draw_box=False):
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

            # box_features_ = []
            # for (img_boxes,feature_map) in zip(nms_boxes,features['res4']):
            #     order = sort_boxes(img_boxes)
            #     each_img_boxes_features = roi_cut(feature_map,img_boxes,order) # implement roi cut!
            #     box_features_.append(each_img_boxes_features)
            # box_features_=torch.cat(box_features_)
            # if self.draw_box:
            #     write_boxes_imgs(nms_boxes, image_Input)

            # [n,1024,14,14] pooler.py line ~220
            box_features_ = self.model.roi_heads.pooler(features_, nms_boxes)
        if self.mode == 'train':
            self.model.train()
        # features of all 1k candidates [n*1000, parameter.box_feat_dim,7,7]
        box_features = self.model.roi_heads.res5(box_features_)
        # [n*1000, parameter.box_feat_dim] need to mean this value
        box_features = box_features.mean(dim=[2, 3])
        return nms_boxes, box_features, eachimg_selected_box_nums, activation

    # im1 = cv2.imread('./imgs/4.jpg')
    # image = im1
    # height, width = image.shape[:2]
    # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    # # print(im1.shape,np.array(image.permute(1, 2, 0)).shape) #(426, 640, 3) (426, 640, 3)

    # inputs = [{"image": image, "height": height, "width": width}]
    # im2 = cv2.imread('./imgs/5.jpg')
    # image = im2
    # height, width = image.shape[:2]
    # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    # inputs.append({"image": image, "height": height, "width": width})
    # im3 = cv2.imread('./imgs/1.jpg')
    # image = im3
    # height, width = image.shape[:2]
    # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    # inputs.append({"image": image, "height": height, "width": width})


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
            box_features[inds:i + inds] = box_features[inds:i + inds] + self.positions[ind]
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
                nn.LayerNorm(emb_size),
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
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


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


class CoS_objects_Classifier(nn.Module):
    def __init__(self, max_num, box_feat_dim):
        super(CoS_objects_Classifier, self).__init__()
        self.det_net = RPNet()
        self.trans_encoder = TransformerEncoder(depth=8, emb_size=box_feat_dim)
        self.pos_e = PosEmbedding(max_num, box_feat_dim)
        self.box_feat_dim = box_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.box_feat_dim, self.box_feat_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.box_feat_dim),
            nn.Dropout(),
            nn.Linear(self.box_feat_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.Linear(1024, 1),
        )
        self.layerNorm = nn.BatchNorm1d(self.box_feat_dim)

    def forward(self, images):
        nms_boxes, box_features, eachimg_selected_box_nums, activation = self.det_net(images)
        # box_features = self.layerNorm(box_features)

        box_features = self.pos_e(eachimg_selected_box_nums, box_features)
        # box_features = self.trans_encoder_init(box_features.reshape(1,-1,parameter.box_feat_dim))
        att_features = self.trans_encoder(box_features.reshape(1, -1, self.box_feat_dim)).reshape(-1, self.box_feat_dim)
        # print(len(box_features),len(att_features),len(eachimg_selected_box_nums),box_features[0].shape,att_features[0].shape)
        # print(att_features,att_features.shape)
        pred_vector = self.classifier(att_features)
        if math.isnan(pred_vector[0][0]):
            print("box_features: ", box_features)
            print("att_features: ", att_features)
            print("nan, value exploded")
            assert False
            # "if appears nan, try to set a smaller lr"
        return nms_boxes, pred_vector


class CoS_Det_Net(nn.Module):
    def __init__(self, cfg, draw_box=False):
        super(CoS_Det_Net, self).__init__()
        self.cfg = cfg
        self.det_net = RPNet(mode='train',
                             backbone=cfg.MODEL.DETECTOR.BACKBONE,
                             backbone_path=cfg.MODEL.DETECTOR.PRETRAINED_PATH,
                             draw_box=draw_box)
        self.box_feat_dim = self.cfg.SOLVER.BOX_FEATURE_DIM
        self.trans_encoder = TransformerEncoder(depth=4, emb_size=self.box_feat_dim )
        self.pos_e = PosEmbedding(cfg.DATA.MAX_NUM, self.box_feat_dim )
        self.fpn = FPN()
        self.score_Layer = ScoreLayer(256)
        self.squeeze_features = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, self.box_feat_dim),
            nn.BatchNorm1d(self.box_feat_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.box_feat_dim, self.box_feat_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.box_feat_dim),
            nn.Dropout(),
            nn.Linear(self.box_feat_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.Linear(1024, 1),
        )
        self.layerNorm = nn.BatchNorm1d(self.box_feat_dim)

    def forward(self, images):
        nms_boxes, box_features, eachimg_selected_box_nums, activation = self.det_net(images)
        box_features = self.squeeze_features(box_features)
        box_features = self.pos_e(eachimg_selected_box_nums, box_features)
        att_features = self.trans_encoder(box_features.reshape(1, -1, self.box_feat_dim)).reshape(-1, self.box_feat_dim)
        pred_vector = self.classifier(att_features)
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
