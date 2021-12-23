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
print(model)


if __name__ == '__main__':

    train_loader = get_loader(img_root, img_size, batch_size, gt_root, max_num=max_num, mode='train', num_thread=1,
                              pin=False)

    # whole_iter_num = 0
    # iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    # for epoch in range(epochs):

        # print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # print('epoch:{0}-------lr:{1}'.format(epoch + 1, lr))

        # epoch_total_loss = 0
        # epoch_loss = 0
    
    for i, data_batch in enumerate(train_loader):
        images = []
        cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
        gts = Variable(data_batch[1].squeeze(0).cuda())
        gt_128 = Variable(data_batch[2].squeeze(0).cuda())
        gt_64 = Variable(data_batch[3].squeeze(0).cuda())
        gt_32 = Variable(data_batch[4].squeeze(0).cuda())
        # print('inputs: ', inputs, inputs.shape)
        # print('gts: ', gts.shape)
        # print('gt_32: ', gt_32,gt_32.shape)

        # torch.set_printoptions(threshold = 10_000)
        # print(gts[0][0].flatten().sum())
        for img in cos_imgs_set:
            images.append({"image": img, "height": img_size, "width": img_size})

        net = RPNet()
        net.cuda()
        nms_boxes,box_features,eachimg_selected_box_nums,activation = net(images)
        # print(box_features,eachimg_selected_box_nums,activation)

        # Norm the object embedding by num of objs in each img
        # for i in range(len(box_features)):
        #     box_features[i] = box_features[i]/eachimg_selected_box_nums[i]
        # att = MultiHeadAttention()
        # att_features = obj_self_atten(box_features,eachimg_selected_box_nums,att)
        # # print(len(box_features),len(att_features),len(eachimg_selected_box_nums),box_features[0].shape,att_features[0].shape)
        # clsfier = CoS_Classifier()
        # clsfier.cuda()
        # out = clsfier(att_features)
        boxes_to_gts_list = boxes_to_gt(nms_boxes, gts)

        
        print(boxes_to_gts_list[:3])
        break
    # gt = gt_32[0][0]
    # torch.set_printoptions(threshold = 10_000)
    # print(gt)

