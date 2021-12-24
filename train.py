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
import parameter
import datetime
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):

    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer

def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_classifier(model):

    train_loader = get_loader(img_root, img_size, batch_size, gt_root, max_num=max_num, mode='train', num_thread=1,
                              pin=False)

    print('''
    Starting training:
        Train steps: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
    '''.format(train_steps, batch_size, parameter.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * batch_size

    # optimizer = optim.SGD([{'params': base_params, 'lr': lr * 0.1},
    #                        {'params': other_params}], lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.BCEWithLogitsLoss() # 1
    # criterion = nn.CrossEntropyLoss(torch.Tensor([0.45,0.55]).cuda()) # 2
    optimizer = optim.Adam(model.parameters(), lr=parameter.lr, weight_decay=0.0001) #86,109,998 parameters

    def print_paras(model):
        for p in model.parameters():
            if p.requires_grad:
                print(p.numel())
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print_paras(model)
    
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
  
    for epoch in range(epochs):

        print('Starting epoch {}/{}----lr:{}.'.format(epoch + 1, epochs, parameter.lr))

        epoch_loss = 0
        epoch_correct_prd_num = 0.0001
        epoch_total_objs_num = 0.0001
        epoch_tot_p = 0.0001
        epoch_true_p = 0.0001
        epoch_tot_p_pred = 0.0001
        epoch_ioa = 0
        for i, data_batch in enumerate(train_loader): # typical total objs num: 15500, positive objs num: 7200
            if (i + 1) > iter_num: break
            if whole_iter_num%200==0: parameter.draw_box = True
            else: parameter.draw_box = False
            cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
            gts = Variable(data_batch[1].squeeze(0).cuda())
            gt_128 = Variable(data_batch[2].squeeze(0).cuda())
            gt_64 = Variable(data_batch[3].squeeze(0).cuda())
            gt_32 = Variable(data_batch[4].squeeze(0).cuda())
            images = []
            for img in cos_imgs_set:
                images.append({"image": img, "height": img_size, "width": img_size})

            optimizer.zero_grad()
            nms_boxes,pred_vector,_ = model(images)  
            # print(pred_vector,pred_vector.shape)
            
            boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts),[])
            # print(boxes_to_gts_list)
            boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).float().cuda() # 1
            # boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).long().cuda() # 2
            
            cls_loss = criterion(pred_vector,boxes_to_gts_list.unsqueeze(1))  # 1
            # cls_loss = criterion(pred_vector,boxes_to_gts_list)  # 2

            obj_num = len(pred_vector)
            correct_pred,y_pred_tags = binary_correct_pred_num(pred_vector,boxes_to_gts_list) #1
            # correct_pred,y_pred_tags = correct_pred_num(pred_vector,boxes_to_gts_list) #2
            cls_loss.backward()
            # print(boxes_to_gts_list)
            # print(pred_vector)
            # print(y_pred_tags.flatten())
            # print(correct_pred)
            # plot_grad_flow_v2(model.named_parameters())
            optimizer.step()
            epoch_loss += cls_loss.cpu().data.item()
            epoch_correct_prd_num += correct_pred.sum().cpu().data.item()
            epoch_total_objs_num+=obj_num

            epoch_tot_p_pred+=y_pred_tags.sum().cpu().data.item()
            epoch_tot_p+=boxes_to_gts_list.sum().cpu().data.item()
            true_p_list = (correct_pred + boxes_to_gts_list>1).float() # ture positive indexes are flagged as 1
            epoch_true_p+=true_p_list.sum().cpu().data.item()
            _,gts_pos_area = boxes_gt_ioa(nms_boxes, gts, pred_vector)

            epoch_ioa+=sum(gts_pos_area).cpu().data.item()/len(gts_pos_area)
            whole_iter_num+=1
            # if np.isnan(epoch_ioa):
            #     print(gts_pos_area)
            #     break
        epoch_loss = epoch_loss/len(train_loader)
        epoch_ioa = epoch_ioa/len(train_loader)


        print('epoch loss: {0:.4f} --- acc: {1:.3f} --- recall: {2:.3f} --- precision: {3:.3f} --- avg.ioa: {4:.3f}'.format(
            epoch_loss,  epoch_correct_prd_num/epoch_total_objs_num,epoch_true_p/epoch_tot_p,epoch_true_p/epoch_tot_p_pred,epoch_ioa))

        if (epoch+1) % 50==0:
            parameter.lr = parameter.lr*lr_decay_gamma
            optimizer = adjust_learning_rate(optimizer, decay_rate=lr_decay_gamma)
        
        
    torch.save(model.state_dict(), save_model_dir + 'cls_epochs{}.pth'.format(epochs))


def train_seg(model):

    train_loader = get_loader(img_root, img_size, batch_size, gt_root, max_num=max_num, mode='train', num_thread=1,
                              pin=False)

    print('''
    Starting training:
        Train steps: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
    '''.format(train_steps, batch_size, parameter.lr, len(train_loader.dataset)))

    criterion = nn.BCEWithLogitsLoss() # 1
    # criterion = nn.CrossEntropyLoss(torch.Tensor([0.45,0.55]).cuda()) # 2
    optimizer = optim.Adam(model.parameters(), lr=parameter.lr, weight_decay=0.0001) #86,109,998 parameters

    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    # epochs = 2
    for epoch in range(epochs):

        print('Starting epoch {}/{}----lr:{}.'.format(epoch + 1, epochs, parameter.lr))

        epoch_loss = 0
      
        for i, data_batch in enumerate(train_loader): # typical total objs num: 15500, positive objs num: 7200
            if (i + 1) > iter_num: break
            if whole_iter_num%200==0: parameter.draw_box = True
            else: parameter.draw_box = False
            cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
            gts = Variable(data_batch[1].squeeze(0).cuda())

            images = []
            for img in cos_imgs_set:
                images.append({"image": img, "height": img_size, "width": img_size})
            optimizer.zero_grad()

            
            nms_boxes,_,output_binary = model(images)  
            boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts),[])
            boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).long() # 2
            pos_imgs_boxes,gts_pos_area = boxes_gt_ioa(nms_boxes, gts, boxes_to_gts_list)

            output_binary = binary_after_boxes(output_binary, pos_imgs_boxes)

            gt_b_maps = binary_after_boxes(gts,pos_imgs_boxes)
            if parameter.draw_box:
                # draw_gt_with_RPboxes(pos_imgs_boxes,gt_b_maps,'binbox')
                draw_gt_with_RPboxes(pos_imgs_boxes,torch.round(torch.sigmoid(output_binary.detach())),'pred_binbox')
            cls_loss = criterion(output_binary.flatten(),gt_b_maps.flatten()) 
            cls_loss.backward()
            epoch_loss += cls_loss.cpu().data.item()
            optimizer.step()
            whole_iter_num+=1
            
        epoch_loss = epoch_loss/len(train_loader)
        
        print('epoch loss: {0:.4f} '.format(
            epoch_loss))
        
        if (epoch+1) % 50==0:
            parameter.lr = parameter.lr*lr_decay_gamma
            optimizer = adjust_learning_rate(optimizer, decay_rate=lr_decay_gamma)
    torch.save(model.state_dict(), save_model_dir + 'seg_epochs{}.pth'.format(epochs))




def train_joint(model):

    train_loader = get_loader(img_root, img_size, batch_size, gt_root, max_num=max_num, mode='train', num_thread=1,
                              pin=False)

    print('''
    Starting training:
        Train steps: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
    '''.format(train_steps, batch_size, parameter.lr, len(train_loader.dataset)))

    # optimizer = optim.SGD([{'params': base_params, 'lr': lr * 0.1},
    #                        {'params': other_params}], lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.BCEWithLogitsLoss() # 1
    # criterion = nn.CrossEntropyLoss(torch.Tensor([0.45,0.55]).cuda()) # 2
    optimizer = optim.Adam(model.parameters(), lr=parameter.lr, weight_decay=0.0001) #86,109,998 parameters

    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    # epochs = 2
    for epoch in range(epochs):

        print('Starting epoch {}/{}----lr:{}.'.format(epoch + 1, epochs, parameter.lr))

        epoch_cls_loss= 0
        epoch_seg_loss= 0
        epoch_correct_prd_num = 0.0001
        epoch_total_objs_num = 0.0001
        epoch_tot_p = 0.0001
        epoch_true_p = 0.0001
        epoch_tot_p_pred = 0.0001
        epoch_ioa = 0
        for i, data_batch in enumerate(train_loader): # typical total objs num: 15500, positive objs num: 7200
            if (i + 1) > iter_num: break
            if whole_iter_num%200==0: parameter.draw_box = True
            else: parameter.draw_box = False
            cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
            gts = Variable(data_batch[1].squeeze(0).cuda())
            # gt_128 = Variable(data_batch[2].squeeze(0).cuda())
            # gt_64 = Variable(data_batch[3].squeeze(0).cuda())
            # gt_32 = Variable(data_batch[4].squeeze(0).cuda())
            images = []
            for img in cos_imgs_set:
                images.append({"image": img, "height": img_size, "width": img_size})
            
            optimizer.zero_grad()
            nms_boxes,pred_vector,output_binary = model(images)  
            
            
            boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts),[])
            boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).float().cuda() # 1
            
            pos_imgs_boxes,gts_pos_area = boxes_gt_ioa(nms_boxes, gts, pred_vector)

            output_binary = binary_after_boxes(output_binary, pos_imgs_boxes)

            gt_b_maps = binary_after_boxes(gts,pos_imgs_boxes)
            if parameter.draw_box:
                # draw_gt_with_RPboxes(pos_imgs_boxes,gt_b_maps,'binbox')
                draw_gt_with_RPboxes(pos_imgs_boxes,torch.round(torch.sigmoid(output_binary.detach())),'pred_binbox')

            cls_loss = criterion(pred_vector,boxes_to_gts_list.unsqueeze(1))  # 1
            seg_loss = criterion(output_binary.flatten(),gt_b_maps.flatten()) 
            loss = seg_loss+cls_loss
            loss.backward()
            optimizer.step()
            epoch_cls_loss += cls_loss.cpu().data.item()
            epoch_seg_loss += seg_loss.cpu().data.item()

          
            obj_num = len(pred_vector)
            correct_pred,y_pred_tags = binary_correct_pred_num(pred_vector,boxes_to_gts_list) #1
     
            epoch_correct_prd_num += correct_pred.sum().cpu().data.item()
            epoch_total_objs_num+=obj_num
            epoch_tot_p_pred+=y_pred_tags.sum().cpu().data.item()
            epoch_tot_p+=boxes_to_gts_list.sum().cpu().data.item()
            true_p_list = (correct_pred + boxes_to_gts_list>1).float() # ture positive indexes are flagged as 1
            epoch_true_p+=true_p_list.sum().cpu().data.item()

            epoch_ioa+=sum(gts_pos_area).cpu().data.item()/len(gts_pos_area)
            whole_iter_num+=1
        
        epoch_cls_loss = epoch_cls_loss/len(train_loader)
        epoch_seg_loss = epoch_seg_loss/len(train_loader)
        epoch_ioa = epoch_ioa/len(train_loader)


        print('epoch class loss: {0:.4f} --- acc: {1:.3f} --- recall: {2:.3f} --- precision: {3:.3f} --- avg.ioa: {4:.3f} --- seg loss: {5:.3f} '.format(
            epoch_cls_loss,  epoch_correct_prd_num/epoch_total_objs_num,epoch_true_p/epoch_tot_p,epoch_true_p/epoch_tot_p_pred,epoch_ioa, epoch_seg_loss))

        if (epoch+1) % 50==0:
            parameter.lr = parameter.lr*lr_decay_gamma
            optimizer = adjust_learning_rate(optimizer, decay_rate=lr_decay_gamma)
    crt_time = datetime.datetime.now()
    torch.save(model.state_dict(), save_model_dir +str(crt_time.month)+'-'+str(crt_time.day)+'-'+str(crt_time.hour)
                    + 'joint_epochs{}.pth'.format(epochs))




if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model = CoS_Det_Net()
    model.train()
    model.cuda()
    # train_seg(model)
    # train_classifier(model)
    train_joint(model)
