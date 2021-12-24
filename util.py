from detectron2.structures import *
import torch
import numpy as np
import cv2
from parameter import *
from self_attention import *
import torch.nn as nn
from torchvision.ops import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.nn.functional as F
from detectron2.structures import Boxes
import parameter
   
def obj_self_atten(box_features,eachimg_selected_box_nums,model):
    obj_embs = []
    ind = 0
    emb_len = box_features.shape[-1]
    att = model
    atts = []

    for obj_num in eachimg_selected_box_nums:
        img_obj = box_features[ind:ind+obj_num]
        obj_embs.append(img_obj)
        assert len(img_obj) == obj_num
        ind+=obj_num
    for inds in range(len(eachimg_selected_box_nums)):
        q = obj_embs[inds].reshape(1,-1,emb_len)
        k = torch.cat([x for i,x in enumerate(obj_embs) if i!=inds]).reshape(1,-1,emb_len)
        atts.append(att(q,k,k).reshape(-1,emb_len))

    atts = torch.cat(atts)
    return (atts)


def sort_boxes(boxes):
    '''
        return a list of index of boxes, sorted with ioa.
        e.g. [3,0,1,2]: boxes[3] has largest ioa, then boxes[0], and so on.
    '''
    pairwise_ioa_mat = pairwise_ioa(boxes,boxes)  # implement for pairwise_ioa in detectron2.structures.boxes
    order = pairwise_ioa_mat.sum(dim=0).argsort(descending=True)
    return(order)

# print(sort_boxes(t))

def filter_boxes_by_prob(boxes_probs,min_num=3,max_mun=40):
    selected_boxes_probs = []
    for each_img_boxes in boxes_probs:
        each_img_boxes = each_img_boxes[:max_mun]
        res = list(filter(lambda x:x>0,each_img_boxes))
        if len(res) <min_num:
            selected_boxes_probs.append(each_img_boxes[:min_num])
        else:
            selected_boxes_probs.append(torch.stack(res))
    return selected_boxes_probs

def draw_gt_with_RPboxes(imgs_boxes, gts, name = 'gt'):
    
    for ix, boxes in enumerate(imgs_boxes):
        im = np.array(gts[ix].repeat(3,1,1).permute(1, 2, 0).cpu()).copy()*255
        # print("print(gt.shape)",im.shape)
        if len(boxes) == 0:
            cv2.imwrite('./RPN_imgs/'+name+str(ix)+'.png',im)
            continue
        for box in boxes:
            cv2.rectangle(im,(box[0].int().item(),box[1].int().item()),(box[2].int().item(),box[3].int().item()),(0,255,0))
        cv2.imwrite('./RPN_imgs/'+name+str(ix)+'.png',im)


def write_boxes_imgs(nms_boxes,inputs):
    for ix, boxes in enumerate(nms_boxes):
        im = np.array(inputs[ix]['image'].permute(1, 2, 0).cpu()).copy()
        # print("print(im.shape)",im.shape)
        for box in boxes:
            cv2.rectangle(im,(box[0].int().item(),box[1].int().item()),(box[2].int().item(),box[3].int().item()),(0,255,0))
        cv2.imwrite('./RPN_imgs/'+str(ix)+'.png',im)


def correct_pred_num(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)  
    # print(y_pred_tags)  
    correct_pred = (y_pred_tags == y_test).float()
    # print(correct_pred)
    return correct_pred,y_pred_tags


def binary_correct_pred_num(y_pred, y_test):
    y_pred_tags = torch.round(torch.sigmoid(y_pred)).flatten()
    correct_pred = (y_pred_tags == y_test).float()
    return correct_pred.flatten(),y_pred_tags

def roi_cut(feature_map_,img_boxes,order):
    feature_map = feature_map_.clone()[None,:]
    rois = []
    for ord in order:
        box = img_boxes[ord.item()]
        roi = roi_align(feature_map,[box.tensor],(14,14),1/16)
        rois.append(roi)
        box = (box.tensor/16).round()[0].int()
        feature_map[0,:,box[1]:box[3],box[0]:box[2]] = feature_map[0,:,box[1]:box[3],box[0]:box[2]]*0
    rois = torch.cat(rois)
    return rois



def roi_cut_boxes_to_gt(imgs_boxes, gts_):
    boxes_to_gts_list = []
    gts = gts_.clone() #size: [8, 1, 256, 256] 8 is image num
    if draw_box:
        draw_gt_with_RPboxes(imgs_boxes, gts)

    for (boxes, gt_) in zip(imgs_boxes, gts):
        order = sort_boxes(boxes)
        gt = gt_[0]
        assert gt.shape[0]==gt.shape[1]
        boxes_to_gt_list = []
        for ind in order:
            gt_area = gt.flatten().sum()
            if gt_area<=0:
                boxes_to_gt_list.append(0)
                continue
            box = boxes[ind.item()].tensor[0]
            assert box.shape == torch.Size([4])
            box = box.round().int()
            [x1,y1,x2,y2] = box
            box_cut_gt = gt[y1:y2+1,x1:x2+1]
            box_gt_area = box_cut_gt.flatten().sum()
            gt[y1:y2+1,x1:x2+1] = gt[y1:y2+1,x1:x2+1]*0
            
            if box_gt_area/ gt_area < 0.2:
                if box_gt_area/((box[2] - box[0]) * (box[3] - box[1]))<0.2:
                    boxes_to_gt_list.append(0)
                else:
                    boxes_to_gt_list.append(1) 
            else:
                boxes_to_gt_list.append(1)
        boxes_to_gt_list = [x for _,x in sorted(zip(order,boxes_to_gt_list))]
        boxes_to_gts_list.append(boxes_to_gt_list)
    assert len(boxes_to_gts_list) == len(imgs_boxes)
    return boxes_to_gts_list


def boxes_to_gt(imgs_boxes, gts_):
    boxes_to_gts_list = []
    gts = gts_.clone() #size: [8, 1, 256, 256] 8 is image num
    # if draw_box:
    #     draw_gt_with_RPboxes(imgs_boxes, gts)
    for (boxes, gt_) in zip(imgs_boxes, gts):
        gt = gt_[0]
        assert gt.shape[0]==gt.shape[1]
        boxes_to_gt_list = []
        gt_area = gt.flatten().sum()
        for box in boxes:
            if gt_area<=0:
                boxes_to_gt_list.append(0)
                continue
            # box = box.tensor
            assert box.shape == torch.Size([4])
            box = box.round().int()
            [x1,y1,x2,y2] = box
            box_cut_gt = gt[y1:y2+1,x1:x2+1]
            box_gt_area = box_cut_gt.flatten().sum()
            
            if box_gt_area/ gt_area < 0.2:
                if box_gt_area/((box[2] - box[0]) * (box[3] - box[1]))<0.2:
                    boxes_to_gt_list.append(0)
                else:
                    boxes_to_gt_list.append(1) 
            else:
                boxes_to_gt_list.append(1)
        boxes_to_gts_list.append(boxes_to_gt_list)
    assert len(boxes_to_gts_list) == len(imgs_boxes)
    return boxes_to_gts_list

def boxes_gt_ioa(imgs_boxes, gts_, pred_vector, at_least_pred_one = True):
    gts = gts_.clone() #size: [8, 1, 256, 256] 8 is image num
    pos_imgs_boxes = []
    gts_pos_area = []
    y_pred_prb = torch.sigmoid(pred_vector).flatten()
    y_pred_tags = torch.round(y_pred_prb)
    boxes_ind = 0

    for (boxes, gt_) in zip(imgs_boxes, gts):
        gt = gt_[0]
        assert gt.shape[0]==gt.shape[1]
        gt_area = gt.flatten().sum()
        pos_boxes = []
        gt_pos_area = 0
        temp_box = boxes[0].tensor[0].round().int()
        temp_max_prob_box = boxes_ind
        for box in boxes:
            box = box.round().int()

            if y_pred_prb[boxes_ind]>y_pred_prb[temp_max_prob_box]:
                temp_box = box
                temp_max_prob_box = boxes_ind
            if y_pred_tags[boxes_ind] == 0:
                boxes_ind+=1
                continue
      
            [x1,y1,x2,y2] = box
            box_cut_gt = gt[y1:y2+1,x1:x2+1]
            box_gt_area = box_cut_gt.flatten().sum()
            gt[y1:y2+1,x1:x2+1] = gt[y1:y2+1,x1:x2+1]*0
            gt_pos_area+=box_gt_area
         
            pos_boxes.append(box)
            boxes_ind+=1
        if len(pos_boxes)==0:
            if at_least_pred_one:
                pos_boxes.append(temp_box)
        if gt_area <= 0:
            gts_pos_area.append(0)
        else:
            gts_pos_area.append(gt_pos_area/gt_area)
        pos_imgs_boxes.append(pos_boxes)

    if parameter.draw_box:
        draw_gt_with_RPboxes(pos_imgs_boxes, gts_, name = 'pred')
    return pos_imgs_boxes,gts_pos_area

def binary_after_boxes(bin_maps_,imgs_boxes, out_size = img_size):
    bin_maps = bin_maps_.clone()
    binary_maps = []
    for (boxes, b_map) in zip(imgs_boxes, bin_maps):
        box_map = torch.zeros(img_size,img_size).cuda()

        if len(boxes) == 0:
            binary_maps.append(b_map*box_map)
            continue
        for box in  boxes:
            [x1,y1,x2,y2] = box
            box_map[y1:y2+1,x1:x2+1] = 1

        binary_maps.append(b_map*box_map)
    return torch.cat(binary_maps)
