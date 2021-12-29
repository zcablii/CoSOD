import argparse

from torch.autograd import Variable
from torch import optim, nn
import torch

from CoSOD.utils.util import AverageMeter, boxes_to_gt, boxes_gt_ioa, binary_correct_pred_num, draw_gt_with_RPboxes, \
    binary_after_boxes, boxes_preded
from CoSOD.dataset.dataset import get_loader
from CoSOD.model import CoS_Det_Net
from CoSOD.utils.evaluator import Evaluator
from configs import cfg

import datetime
import math
import time
import os

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='PyTorch SOD FOR CSNet')
parser.add_argument(
    "--config",
    default="configs/cosal-joint-train.yml",
    metavar="FILE",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)

if cfg.GPU.USE:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in cfg.GPU.ID)

if cfg.TASK == '':
    cfg.TASK = cfg.MODEL.ARCH

time_now = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
train_log_name = 'train_log_' + cfg.TASK + "_" + time_now + '.txt'
val_log_name = 'val_log_' + cfg.TASK + "_" + time_now + '.txt'
check_point_dir = os.path.join(cfg.MODEL.SAVEDIR, cfg.TASK, 'checkpoint' + "_" + time_now)
logging_dir = os.path.join(cfg.MODEL.SAVEDIR, cfg.TASK)
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
if not os.path.isdir(check_point_dir):
    os.mkdir(check_point_dir)
TRAIN_LOG_FOUT = open(os.path.join(logging_dir, train_log_name), 'w')
VAL_LOG_FOUT = open(os.path.join(logging_dir, val_log_name), 'w')


def train_log_string(out_str, display=True):
    out_str = str(out_str)
    TRAIN_LOG_FOUT.write(out_str + '\n')
    TRAIN_LOG_FOUT.flush()
    if display:
        print(out_str)


def val_log_string(out_str, display=False):
    out_str = str(out_str)
    VAL_LOG_FOUT.write(out_str + '\n')
    VAL_LOG_FOUT.flush()
    if display:
        print(out_str)


train_log_string(cfg)


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        train_log_string('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        train_log_string('after lr: ', param_group['lr'])
    return optimizer


def train_classifier(model, train_loader, epoch, optimizer, criterion):
    epoch_loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ioa = AverageMeter()
    epoch_correct_prd_num = 0.0001
    epoch_total_objs_num = 0.0001
    epoch_tot_p = 0.0001
    epoch_true_p = 0.0001
    epoch_tot_p_pred = 0.0001
    end = time.time()
    max_iterate = math.ceil(len(train_loader.dataset) / cfg.DATA.BATCH_SIZE)
    # typical total objs num: 15500, positive objs num: 7200
    for i, data_batch in enumerate(train_loader):
        draw_box = cfg.LOG.DRAW_BOX and (i + (epoch + 1) * len(train_loader)) % cfg.LOG.EPOCH_FREQ == 0
        if (i + 1) > max_iterate: break
        data_time.update(time.time() - end)
        cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
        gts = Variable(data_batch[1].squeeze(0).cuda())

        images = []
        for img in cos_imgs_set:
            images.append({"image": img,
                           "height": cfg.DATA.IMAGE_H,
                           "width": cfg.DATA.IMAGE_W})

        optimizer.zero_grad()
        nms_boxes, pred_vector, _ = model(images)
        # print(pred_vector.shape)

        boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts), [])
        # print(boxes_to_gts_list)
        boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).float().cuda()  # 1
        # boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).long().cuda() # 2

        cls_loss = criterion(pred_vector, boxes_to_gts_list.unsqueeze(1))  # 1
        # cls_loss = criterion(pred_vector,boxes_to_gts_list)  # 2

        obj_num = len(pred_vector)
        correct_pred, y_pred_tags = binary_correct_pred_num(pred_vector, boxes_to_gts_list)  # 1
        # correct_pred,y_pred_tags = correct_pred_num(pred_vector,boxes_to_gts_list) #2
        cls_loss.backward()
        # print(boxes_to_gts_list)
        # print(pred_vector)
        # print(y_pred_tags.flatten())
        # print(correct_pred)
        # plot_grad_flow_v2(model.named_parameters())
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        epoch_loss.update(cls_loss.cpu().data.item())
        epoch_correct_prd_num += correct_pred.sum().cpu().data.item()
        epoch_total_objs_num += obj_num
        epoch_tot_p_pred += y_pred_tags.sum().cpu().data.item()
        epoch_tot_p += boxes_to_gts_list.sum().cpu().data.item()
        true_p_list = (correct_pred + boxes_to_gts_list > 1).float()  # ture positive indexes are flagged as 1
        epoch_true_p += true_p_list.sum().cpu().data.item()

        _, gts_pos_area = boxes_gt_ioa(nms_boxes, gts, pred_vector, False, draw_box)

        epoch_ioa.update(sum(gts_pos_area).cpu().data.item() / len(gts_pos_area))
        # if i % cfg.LOG.BATCH_FREQ == 0 and i != 0:
        #     train_log_string('ClassEpoch: [{0}][{1}/{2}]\t'
        #                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                      .format(epoch, i, len(train_loader),
        #                              batch_time=batch_time,
        #                              data_time=data_time,
        #                              loss=epoch_loss))

    if (epoch + 1) % cfg.LOG.EPOCH_FREQ == 0 or epoch == 0:
        train_log_string('classification --- '
                         'epoch: {0:d} --- '
                         'loss: {1:.4f} --- '
                         'acc: {2:.3f} --- '
                         'recall: {3:.3f} --- '
                         'precision: {4:.3f} --- '
                         'avg.ioa: {5:.3f}'.format(epoch,
                                                   epoch_loss.avg,
                                                   epoch_correct_prd_num / epoch_total_objs_num,
                                                   epoch_true_p / epoch_tot_p,
                                                   epoch_true_p / epoch_tot_p_pred,
                                                   epoch_ioa.avg))


def train_seg(model, train_loader, epoch, optimizer, criterion):
    epoch_loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    max_iterate = math.ceil(len(train_loader.dataset) / cfg.DATA.BATCH_SIZE)
    # typical total objs num: 15500, positive objs num: 7200
    for i, data_batch in enumerate(train_loader):
        draw_box = cfg.LOG.DRAW_BOX and (i + (epoch + 1) * len(train_loader)) % cfg.LOG.EPOCH_FREQ == 0
        if (i + 1) > max_iterate: break

        data_time.update(time.time() - end)
        cos_imgs_set = Variable(data_batch[0].squeeze(0).cuda())
        gts = Variable(data_batch[1].squeeze(0).cuda())

        images = []
        for img in cos_imgs_set:
            images.append({"image": img,
                           "height": cfg.DATA.IMAGE_H,
                           "width": cfg.DATA.IMAGE_W})

        optimizer.zero_grad()
        nms_boxes, _, output_binary = model(images)
        boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts), [])
        boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).long()  # 2
        pos_imgs_boxes, gts_pos_area = boxes_gt_ioa(nms_boxes, gts, boxes_to_gts_list, False)
        output_binary = binary_after_boxes(output_binary, pos_imgs_boxes, (cfg.DATA.IMAGE_H,
                                                                           cfg.DATA.IMAGE_W))

        gt_b_maps = binary_after_boxes(gts, pos_imgs_boxes, (cfg.DATA.IMAGE_H,
                                                             cfg.DATA.IMAGE_W))
        if draw_box:
            draw_gt_with_RPboxes(pos_imgs_boxes, torch.round(torch.sigmoid(output_binary.detach())), 'pred_binbox')
        cls_loss = criterion(output_binary.flatten(), gt_b_maps.flatten())
        cls_loss.backward()
        epoch_loss.update(cls_loss.cpu().data.item())
        optimizer.step()

        # if i % cfg.LOG.BATCH_FREQ == 0 and i != 0:
        #     train_log_string('SegEpoch: [{0}][{1}/{2}]\t'
        #                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                      .format(epoch, i, len(train_loader),
        #                              batch_time=batch_time,
        #                              data_time=data_time,
        #                              loss=epoch_loss))

    if (epoch + 1) % cfg.LOG.EPOCH_FREQ == 0 or epoch == 0:
        train_log_string('segmentation --- '
                         'epoch: {0:d} --- '
                         'loss: {1:.4f}'.format(epoch,
                                                epoch_loss.avg))


def eval_model(model, eval_loader, epoch):
    mae = AverageMeter()
    f_measure = AverageMeter()
    e_measure = AverageMeter()
    s_measure = AverageMeter()
    auc = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ioa = AverageMeter()
    epoch_correct_prd_num = 0.0001
    epoch_total_objs_num = 0.0001
    epoch_tot_p = 0.0001
    epoch_true_p = 0.0001
    epoch_tot_p_pred = 0.0001
    end = time.time()
    model.eval()
    max_iterate = math.ceil(len(eval_loader.dataset) / cfg.DATA.BATCH_SIZE)
    for i, data_batch in enumerate(eval_loader):
        if (i + 1) > max_iterate: break
        imgs_groups = Variable(data_batch[0].squeeze(0).cuda())
        gt_groups = Variable(data_batch[1].squeeze(0).cuda())
        num = imgs_groups.shape[0]
        group_nums = num // cfg.DATA.MAX_NUM + (1 if num % cfg.DATA.MAX_NUM > 0 else 0)
        batch_imgs = []
        batch_gts = []
        for i in range(group_nums - 1):
            batch_imgs.append(imgs_groups[i * cfg.DATA.MAX_NUM:(i + 1) * cfg.DATA.MAX_NUM])
            batch_gts.append(gt_groups[i * cfg.DATA.MAX_NUM:(i + 1) * cfg.DATA.MAX_NUM])
        batch_imgs.append(imgs_groups[-cfg.DATA.MAX_NUM:])
        batch_gts.append(gt_groups[-cfg.DATA.MAX_NUM:])
        for imgs, gts in zip(batch_imgs, batch_gts):
            inputs = []
            for img in imgs:
                inputs.append({"image": img, "height": cfg.DATA.IMAGE_H, "width": cfg.DATA.IMAGE_W})
            with torch.no_grad():
                nms_boxes, pred_vector, output_binary = model(inputs)
                boxes_to_gts_list = sum(boxes_to_gt(nms_boxes, gts), [])
                boxes_to_gts_list = torch.Tensor(boxes_to_gts_list).float().cuda()  # 1
                obj_num = len(pred_vector)
                correct_pred, y_pred_tags = binary_correct_pred_num(pred_vector, boxes_to_gts_list)  # 1
                batch_time.update(time.time() - end)
                end = time.time()
                epoch_correct_prd_num += correct_pred.sum().cpu().data.item()
                epoch_total_objs_num += obj_num
                epoch_tot_p_pred += y_pred_tags.sum().cpu().data.item()
                epoch_tot_p += boxes_to_gts_list.sum().cpu().data.item()
                true_p_list = (correct_pred + boxes_to_gts_list > 1).float()
                epoch_true_p += true_p_list.sum().cpu().data.item()
                _, gts_pos_area = boxes_gt_ioa(nms_boxes, gts, pred_vector, False)
                epoch_ioa.update(sum(gts_pos_area).cpu().data.item() / len(gts_pos_area))
                output_binary = torch.round(torch.sigmoid(output_binary))
                pos_imgs_boxes = boxes_preded(nms_boxes, pred_vector)
                output_binary = binary_after_boxes(output_binary, pos_imgs_boxes, (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W))
                evaluator = Evaluator(list(zip(output_binary, gts)), cfg.GPU.USE)
                mae.update(evaluator.Eval_mae())
                # f_measure.update(evaluator.Eval_fmeasure()[0])
                # e_measure.update(evaluator.Eval_Emeasure())
                # s_measure.update(evaluator.Eval_Smeasure())
                # auc.update(evaluator.Eval_auc()[0])

        # train_log_string('\tValEpoch: [{0}][{1}/{2}]\t'
        #                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                  'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
        #                  'Fmeasure {f_measure.val:.3f} ({f_measure.avg:.3f})\t'
        #                  'Smeasure {s_measure.val:.3f} ({s_measure.avg:.3f})\t'
        #                  'Emeasure {e_measure.val:.3f} ({e_measure.avg:.3f})\t'
        #                  'AUC {auc.val:.3f} ({auc.avg:.3f})'
        #                  .format(epoch, i, len(eval_loader),
        #                          batch_time=batch_time,
        #                          data_time=data_time,
        #                          mae=mae,
        #                          f_measure=f_measure,
        #                          e_measure=e_measure,
        #                          s_measure=s_measure,
        #                          auc=auc
        #                          ))

    train_log_string('\tClassification ValEpoch: {0}\t'
                     'Acc {acc:.3f}\t'
                     'Recall {recall:.3f}\t'
                     'Precision {precision:.3f}\t'
                     'IOA {ioa.avg:.3f}'
                     .format(epoch,
                             acc=epoch_correct_prd_num / epoch_total_objs_num,
                             recall=epoch_true_p / epoch_tot_p,
                             precision=epoch_true_p / epoch_tot_p_pred,
                             ioa=epoch_ioa))
    train_log_string('\tSegmentation ValEpoch: {0}\t'
                     'MAE {mae.avg:.3f}\t'
                     'Fmeasure {f_measure.avg:.3f}\t'
                     'Smeasure {s_measure.avg:.3f}\t'
                     'Emeasure {e_measure.avg:.3f}\t'
                     'AUC {auc.avg:.3f}'
                     .format(epoch,
                             mae=mae,
                             f_measure=f_measure,
                             e_measure=e_measure,
                             s_measure=s_measure,
                             auc=auc))


def main():
    train_loader = get_loader(cfg)
    if cfg.VAL.USE:
        eval_loader = get_loader(cfg, mode="eval")
    else:
        eval_loader = []
    model = CoS_Det_Net(cfg)
    model.train()
    model.cuda()
    train_log_string('Starting training:'
                     '  Train steps: {}'
                     '  Batch size: {}'
                     '  Learning rate: {}'
                     '  Training set size: {}'
                     '  Evaluation set size: {}\n'
                     .format(cfg.SOLVER.TRAIN_STEPS,
                             cfg.DATA.BATCH_SIZE,
                             cfg.SOLVER.LR,
                             len(train_loader.dataset),
                             len(eval_loader.dataset)))

    criterion = nn.BCEWithLogitsLoss()
    # 86,109,998 parameters
    optimizer_classifier = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=0.0001)
    optimizer_seg = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=0.0001)
    iterate = 0
    if cfg.SOLVER.JOINT:
        for epoch in range(cfg.SOLVER.MAX_EPOCHS):
            train_classifier(model=model,
                             train_loader=train_loader,
                             epoch=epoch,
                             optimizer=optimizer_classifier,
                             criterion=criterion)
            train_seg(model=model,
                      train_loader=train_loader,
                      epoch=epoch,
                      optimizer=optimizer_seg,
                      criterion=criterion)

            if (epoch + 1) % cfg.SOLVER.DECAY_STEPS == 0:
                optimizer_classifier = adjust_learning_rate(optimizer_classifier, decay_rate=cfg.SOLVER.LR_DECAY_GAMMA)
                optimizer_seg = adjust_learning_rate(optimizer_seg, decay_rate=cfg.SOLVER.LR_DECAY_GAMMA)
            if (epoch + 1) % cfg.LOG.EPOCH_FREQ != 0:
                if cfg.VAL.USE:
                    eval_model(model, eval_loader, epoch)
                save_file = os.path.join(
                    check_point_dir, 'checkpoint_epoch{}.pth.tar'.format(epoch + 1))
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        "iterate": iterate,
                        'arch': cfg.MODEL.ARCH,
                        'state_dict': model.state_dict(),
                        'optimizer_classifier': optimizer_classifier.state_dict(),
                        'optimizer_seg': optimizer_seg.state_dict()
                    },
                    filename=save_file)
                print("Epoch:{} Save Done".format(epoch + 1))
            iterate += len(train_loader)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == "__main__":
    main()
