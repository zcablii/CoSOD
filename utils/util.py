from detectron2.structures import pairwise_ioa
from torchvision.ops import roi_align
import numpy as np
import torch
import cv2
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def obj_self_atten(box_features, eachimg_selected_box_nums, model):
    obj_embs = []
    ind = 0
    emb_len = box_features.shape[-1]
    att = model
    atts = []

    for obj_num in eachimg_selected_box_nums:
        img_obj = box_features[ind:ind + obj_num]
        obj_embs.append(img_obj)
        assert len(img_obj) == obj_num
        ind += obj_num
    for inds in range(len(eachimg_selected_box_nums)):
        q = obj_embs[inds].reshape(1, -1, emb_len)
        k = torch.cat([x for i, x in enumerate(obj_embs) if i != inds]).reshape(1, -1, emb_len)
        atts.append(att(q, k, k).reshape(-1, emb_len))

    atts = torch.cat(atts)
    return (atts)


def sort_boxes(boxes):
    '''
        return a list of index of boxes, sorted with ioa.
        e.g. [3,0,1,2]: boxes[3] has largest ioa, then boxes[0], and so on.
    '''
    pairwise_ioa_mat = pairwise_ioa(boxes, boxes)  # implement for pairwise_ioa in detectron2.structures.boxes
    order = pairwise_ioa_mat.sum(dim=0).argsort(descending=True)
    return (order)


# print(sort_boxes(t))

def MAE(preds, gts):
    avg_mae, img_num = 0.0, 0.0
    with torch.no_grad():
        for pred, gt in zip(preds, gts):
            mea = torch.abs(pred - gt).mean()
            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
        avg_mae /= img_num
        return avg_mae.item()
        
def filter_boxes_by_prob(boxes_probs, min_num=10, max_mun=10):
    selected_boxes_probs = []
    for each_img_boxes in boxes_probs:
        each_img_boxes = each_img_boxes[:max_mun]
        res = list(filter(lambda x: x > 0, each_img_boxes))
        if len(res) < min_num:
            selected_boxes_probs.append(each_img_boxes[:min_num])
        else:
            selected_boxes_probs.append(torch.stack(res))
    return selected_boxes_probs


def draw_gt_with_RPboxes(imgs_boxes, gts, name='gt', gt_boxes=None, pred_map = None):
    if not gt_boxes is None:
        assert not pred_map is None
        ix = 0
        for pred_boxes, gt_boxes in zip(imgs_boxes, gt_boxes):
            gt_map = np.array(gts[ix].permute(1, 2, 0).cpu()).copy() * 255
            im = np.array(pred_map[ix].repeat(3, 1, 1).permute(1, 2, 0).cpu()).copy() * 255
            im[:,:,0] = gt_map[:,:,0]

            # print("print(gt.shape)",im.shape)
            if len(gt_boxes) == 0 and len(pred_boxes)==0:
                cv2.imwrite('./RPN_imgs/' + name + str(ix) + '.png', im)
                ix+=1
                continue
            if len(pred_boxes) !=0:
                for pred_box in pred_boxes:
                    cv2.rectangle(im, (pred_box[0].int().item(), pred_box[1].int().item()), (pred_box[2].int().item(), pred_box[3].int().item()),
                                (0, 255, 255))
            if len(gt_boxes)!=0:
                for gt_boxe in gt_boxes:
                    
                    cv2.rectangle(im, (gt_boxe[0].int().item()-1, gt_boxe[1].int().item()-1), (gt_boxe[2].int().item()+1, gt_boxe[3].int().item()+1),
                                (255, 0, 0))
            cv2.imwrite('./RPN_imgs/' + name + str(ix) + '.png', im)
            ix+=1


    else :
        for ix, boxes in enumerate(imgs_boxes):
            im = np.array(gts[ix].repeat(3, 1, 1).permute(1, 2, 0).cpu()).copy() * 255
            # print("print(gt.shape)",im.shape)
            if len(boxes) == 0:
                cv2.imwrite('./RPN_imgs/' + name + str(ix) + '.png', im)
                continue
            for box in boxes:
                cv2.rectangle(im, (box[0].int().item(), box[1].int().item()), (box[2].int().item(), box[3].int().item()),
                            (0, 255, 0))
            cv2.imwrite('./RPN_imgs/' + name + str(ix) + '.png', im)


def write_boxes_imgs(nms_boxes, inputs, name=''):
    for ix, boxes in enumerate(nms_boxes):
        im = np.array(inputs[ix]['image'].permute(1, 2, 0).cpu()).copy()
        # print("print(im.shape)",im.shape)
        for box in boxes:
            cv2.rectangle(im, (box[0].int().item(), box[1].int().item()), (box[2].int().item(), box[3].int().item()),
                          (0, 255, 0))
        cv2.imwrite('./RPN_imgs/' +name+ str(ix) + '.png', im)


def correct_pred_num(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    # print(y_pred_tags)  
    correct_pred = (y_pred_tags == y_test).float()
    # print(correct_pred)
    return correct_pred, y_pred_tags


def binary_correct_pred_num(y_pred, y_test):
    y_pred_tags = torch.round(torch.sigmoid(y_pred)).flatten()
    correct_pred = (y_pred_tags == y_test).float()
    return correct_pred.flatten(), y_pred_tags


def roi_cut(feature_map_, img_boxes, order):
    feature_map = feature_map_.clone()[None, :]
    rois = []
    for ord in order:
        box = img_boxes[ord.item()]
        roi = roi_align(feature_map, [box.tensor], (14, 14), 1 / 16)
        rois.append(roi)
        box = (box.tensor / 16).round()[0].int()
        feature_map[0, :, box[1]:box[3], box[0]:box[2]] = feature_map[0, :, box[1]:box[3], box[0]:box[2]] * 0
    rois = torch.cat(rois)
    return rois


def roi_cut_boxes_to_gt(imgs_boxes, gts_, draw_box=False):
    boxes_to_gts_list = []
    gts = gts_.clone()  # size: [8, 1, 256, 256] 8 is image num
    if draw_box:
        draw_gt_with_RPboxes(imgs_boxes, gts)

    for (boxes, gt_) in zip(imgs_boxes, gts):
        order = sort_boxes(boxes)
        gt = gt_[0]
        assert gt.shape[0] == gt.shape[1]
        boxes_to_gt_list = []
        for ind in order:
            gt_area = gt.flatten().sum()
            if gt_area <= 0:
                boxes_to_gt_list.append(0)
                continue
            box = boxes[ind.item()].tensor[0]
            assert box.shape == torch.Size([4])
            box = box.round().int()
            [x1, y1, x2, y2] = box
            box_cut_gt = gt[y1:y2 + 1, x1:x2 + 1]
            box_gt_area = box_cut_gt.flatten().sum()
            gt[y1:y2 + 1, x1:x2 + 1] = gt[y1:y2 + 1, x1:x2 + 1] * 0

            if box_gt_area / gt_area < 0.2:
                if box_gt_area / ((box[2] - box[0]) * (box[3] - box[1])) < 0.2:
                    boxes_to_gt_list.append(0)
                else:
                    boxes_to_gt_list.append(1)
            else:
                boxes_to_gt_list.append(1)
        boxes_to_gt_list = [x for _, x in sorted(zip(order, boxes_to_gt_list))]
        boxes_to_gts_list.append(boxes_to_gt_list)
    assert len(boxes_to_gts_list) == len(imgs_boxes)
    return boxes_to_gts_list


def boxes_to_gt(imgs_boxes, gts_):
    boxes_to_gts_list = []
    gts = gts_.clone()  # size: [8, 1, 256, 256] 8 is image num
    # if draw_box:
    #     draw_gt_with_RPboxes(imgs_boxes, gts)
    for (boxes, gt_) in zip(imgs_boxes, gts):
        gt = gt_[0]
        assert gt.shape[0] == gt.shape[1]
        boxes_to_gt_list = []
        gt_area = gt.flatten().sum()
        for box in boxes:
            if gt_area <= 0:
                boxes_to_gt_list.append(0)
                continue
            # box = box.tensor
            assert box.shape == torch.Size([4])
            box = box.round().int()
            [x1, y1, x2, y2] = box
            box_cut_gt = gt[y1:y2 + 1, x1:x2 + 1]
            box_gt_area = box_cut_gt.flatten().sum()
            bbox_area = ((box[2] - box[0]) * (box[3] - box[1]))

            # if box_gt_area / gt_area < 0.2:
            #     if box_gt_area / bbox_area < 0.2:
            #         boxes_to_gt_list.append(0)
            #     else:
            #         boxes_to_gt_list.append(1)
            # else:
            #     boxes_to_gt_list.append(1)

            # if box_gt_area / gt_area < 0.2 or box_gt_area / bbox_area < 0.2:
            if box_gt_area / gt_area < 0.3 or box_gt_area / bbox_area < 0.2:
            # if box_gt_area/(gt_area+bbox_area-box_gt_area)<0.2:
                    boxes_to_gt_list.append(0)
            else:
                boxes_to_gt_list.append(1)

        boxes_to_gts_list.append(boxes_to_gt_list)
    assert len(boxes_to_gts_list) == len(imgs_boxes)
    return boxes_to_gts_list


def boxes_gt_ioa(imgs_boxes, gts_, pred_vector, at_least_pred_one=True):
    gts = gts_.clone()  # size: [8, 1, 256, 256] 8 is image num
    pos_imgs_boxes = []
    gts_pos_area = []
    y_pred_prb = torch.sigmoid(pred_vector).flatten()
    y_pred_tags = torch.round(y_pred_prb)
    boxes_ind = 0

    for (boxes, gt_) in zip(imgs_boxes, gts):
        gt = gt_[0]
        assert gt.shape[0] == gt.shape[1]
        gt_area = gt.flatten().sum()
        pos_boxes = []
        gt_pos_area = 0
        temp_box = boxes[0].tensor[0].round().int()
        temp_max_prob_box = boxes_ind
        for box in boxes:
            box = box.round().int()

            if y_pred_prb[boxes_ind] > y_pred_prb[temp_max_prob_box]:
                temp_box = box
                temp_max_prob_box = boxes_ind
            if y_pred_tags[boxes_ind] == 0:
                boxes_ind += 1
                continue

            [x1, y1, x2, y2] = box
            box_cut_gt = gt[y1:y2 + 1, x1:x2 + 1]
            box_gt_area = box_cut_gt.flatten().sum()
            gt[y1:y2 + 1, x1:x2 + 1] = gt[y1:y2 + 1, x1:x2 + 1] * 0
            gt_pos_area += box_gt_area

            pos_boxes.append(box)
            boxes_ind += 1
        if len(pos_boxes) == 0:
            if at_least_pred_one:
                pos_boxes.append(temp_box)
        if gt_area <= 0:
            gts_pos_area.append(0)
        else:
            gts_pos_area.append(gt_pos_area / gt_area)
        pos_imgs_boxes.append(pos_boxes)

    return pos_imgs_boxes, gts_pos_area


def binary_after_boxes(bin_maps_, imgs_boxes, img_size):
    bin_maps = bin_maps_.clone()
    binary_maps = []
    for (boxes, b_map) in zip(imgs_boxes, bin_maps):
        box_map = torch.zeros(img_size[0], img_size[1]).cuda()

        if len(boxes) == 0:
            binary_maps.append(b_map * box_map)
            continue
        for box in boxes:
            [x1, y1, x2, y2] = box
            box_map[y1:y2 + 1, x1:x2 + 1] = 1

        binary_maps.append(b_map * box_map)
    return torch.cat(binary_maps).reshape(-1, 1, img_size[0], img_size[1])


def boxes_preded(imgs_boxes, pred_vector, at_least_pred_one=True):
    pos_imgs_boxes = []

    y_pred_prb = torch.sigmoid(pred_vector).flatten()
    y_pred_tags = torch.round(y_pred_prb)
    boxes_ind = 0

    for boxes in imgs_boxes:
        pos_boxes = []
        temp_box = boxes[0].tensor[0].round().int()
        temp_max_prob_box = boxes_ind
        for box in boxes:
            box = box.round().int()
            if y_pred_prb[boxes_ind] > y_pred_prb[temp_max_prob_box]:
                temp_box = box
                temp_max_prob_box = boxes_ind
            if y_pred_tags[boxes_ind] == 0:
                boxes_ind += 1
                continue

            pos_boxes.append(box)
            boxes_ind += 1
        if len(pos_boxes) == 0:
            if at_least_pred_one:
                pos_boxes.append(temp_box)

        pos_imgs_boxes.append(pos_boxes)

    return pos_imgs_boxes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key, negative_keys=None, negative_mode='co2other'):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='co2other'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'co2other' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'co2other'.")
   
    # Check matching number of samples.
 
    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    # query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        if negative_mode == 'co2other':
            positive_logit = torch.mean(query @ transpose(positive_key),dim=1, keepdim=True)
        

        elif negative_mode == 'other2co':
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        
        negative_logits = query @ transpose(negative_keys)
           

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
  
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device) # len = len of query
        # print("2",logits)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



def loss_for_infNCE(boxes_to_gts_list, q):
    emb_size = q.shape[-1]
    tran = nn.Linear(emb_size, emb_size).cuda()
    loss_fun = InfoNCE()
    losses = []
    q = tran(q)

    inds1 = [i for i,x in enumerate(boxes_to_gts_list) if x == 1]
    inds0 = [i for i,x in enumerate(boxes_to_gts_list) if x == 0]

    cos_q = q[inds1]
    other_q = q[inds0]

    loss_co2other = loss_fun(cos_q, cos_q, other_q)
    loss_other2co = loss_fun(other_q, other_q, cos_q,negative_mode='other2co')
    losses.append(loss_co2other+loss_other2co/2)
    return sum(losses)/len(losses)



class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

def loss_for_triplet(boxes_to_gts_list, q):
   
    loss_fun = TripletLoss()
    losses = []

    inds1 = [i for i,x in enumerate(boxes_to_gts_list) if x == 1]
    inds0 = [i for i,x in enumerate(boxes_to_gts_list) if x == 0]
    if len(inds1)<2 or len(inds0)<1:
        return 0
    for ind_a in inds1:
        for ind_p in inds1:
            if ind_a == ind_p:
                continue
            for ind_n in inds0:
                trip_loss = loss_fun(q[ind_a].reshape(1,-1),q[ind_p].reshape(1,-1),q[ind_n].reshape(1,-1))
                losses.append(trip_loss)
    return sum(losses)/len(losses)
