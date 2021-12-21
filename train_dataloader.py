import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from model import *
from dataset import get_loader
import math
from parameter import *
from self_attention import *


model = CoS_objects_Classifier()
# print(model)
model.cuda()

if __name__ == '__main__':

    train_loader = get_loader(img_root, img_size, batch_size, gt_root, max_num=max_num, mode='train', num_thread=1,
                              pin=False)
   
   
    for epoch in range(epochs):

        print('Starting epoch {}/{}----lr:{}.'.format(epoch + 1, epochs, lr))

        epoch_loss = 0
        epoch_correct_prd_num = 0.0001
        epoch_total_objs_num = 0.0001
        epoch_tot_p = 0.0001
        epoch_true_p = 0.0001
        epoch_tot_p_pred = 0.0001
        epoch_ioa = 0
        for i, data_batch in enumerate(train_loader): # typical total objs num: 15500, positive objs num: 7200

            cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
            gts = Variable(data_batch[1].squeeze(0).cuda())

            images = []
            for img in cos_imgs_set:
                images.append({"image": img, "height": img_size, "width": img_size})

            nms_boxes,pred_vector = model(images)  
            # print(pred_vector,pred_vector.shape)
            
            boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts),[])

            boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).long().cuda() # 2
            
            obj_num = len(pred_vector)
            correct_pred,y_pred_tags = correct_pred_num(pred_vector,boxes_to_gts_list)
            
            print(boxes_to_gts_list, y_pred_tags )
            _,gts_pos_area = boxes_gt_ioa(nms_boxes, gts, pred_vector)

            epoch_ioa+=sum(gts_pos_area).cpu().data.item()/len(gts_pos_area)
            print(epoch_ioa)

            break
        # epoch_ioa = epoch_ioa/len(train_loader)

        print('epoch loss: {0:.4f} --- acc: {1:.3f} --- recall: {2:.3f} --- precision: {3:.3f} --- avg.ioa: {4:.3f}'.format(
            epoch_loss,  epoch_correct_prd_num/epoch_total_objs_num,epoch_true_p/epoch_tot_p,epoch_true_p/epoch_tot_p_pred,epoch_ioa))
        break
