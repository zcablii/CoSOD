import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from configs import cfg
from dataset.dataset import *
from model import CoS_Det_Net
from utils.evaluator import Evaluator
from utils.util import *
from model import *
import os
import argparse

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

def test_net(model, batch_size):

    test_loaders = get_loader(cfg, mode='test')
    for dataset_idx in range(len(cfg.TEST.IMAGE_DIRS)):
        dataset_name = cfg.TEST.IMAGE_DIRS[dataset_idx].split('/')[-2]

        print('testing {}'.format(cfg.TEST.IMAGE_DIRS[dataset_idx]))
        test_loader = test_loaders[dataset_idx]
        print('''
            Starting testing:
                Batch size: {}
                Testing size: {}
            '''.format(batch_size, len(test_loader.dataset)))
        iter_num = len(test_loader.dataset) // batch_size
        for i, data_batch in enumerate(test_loader):
            print('{}/{}'.format(i, len(test_loader.dataset)))
            if (i + 1) > iter_num: break

            cos_imgs_groups = Variable(data_batch[0].squeeze(0).cuda())
            group_subpaths = data_batch[2]
            group_ori_sizes = data_batch[3]

            # print(cos_imgs_groups.shape)
            # print(len(group_subpaths),group_subpaths[0])
            num = cos_imgs_groups.shape[0]
            group_nums = num//cfg.DATA.MAX_NUM + (1 if num%cfg.DATA.MAX_NUM>0 else 0)

            batch_groups_paths = []
            batch_imgs = []
            batch_group_ori_sizes = []
            for i in range(group_nums-1):
                batch_groups_paths.append(group_subpaths[i*cfg.DATA.MAX_NUM:(i+1)*cfg.DATA.MAX_NUM])
                batch_imgs.append(cos_imgs_groups[i*cfg.DATA.MAX_NUM:(i+1)*cfg.DATA.MAX_NUM])
                batch_group_ori_sizes.append(group_ori_sizes[i*cfg.DATA.MAX_NUM:(i+1)*cfg.DATA.MAX_NUM])

            batch_groups_paths.append(group_subpaths[-cfg.DATA.MAX_NUM:])
            batch_imgs.append(cos_imgs_groups[-cfg.DATA.MAX_NUM:])
            batch_group_ori_sizes.append(group_ori_sizes[-cfg.DATA.MAX_NUM:])
            for group, subpaths,ori_sizes in zip(batch_imgs, batch_groups_paths,batch_group_ori_sizes):
                # print(group.shape)
                # print(len(subpaths))
                inputs = []
                for img in group:
                    inputs.append({"image": img, "height": cfg.DATA.IMAGE_H, "width": cfg.DATA.IMAGE_H})

                nms_boxes,pred_vector,output_binary = model(inputs)  
                output_binary = torch.round(torch.sigmoid(output_binary))
                pos_imgs_boxes = boxes_preded(nms_boxes,pred_vector)

                output_binary = binary_after_boxes(output_binary, pos_imgs_boxes, (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W))
                output = output_binary
                # print(output)
                saved_root = cfg.TEST.SAVE_PATH + dataset_name
                # save_final_path = saved_root + '/CADC_' + test_model + '/' + subpaths[0][0].split('/')[0] + '/'
                save_final_path = saved_root + '/' + subpaths[0][0].split('/')[0] + '/'
                os.makedirs(save_final_path, exist_ok=True)

                for inum in range(output.size(0)):
                    pre = output[inum, :, :, :].data.cpu()

                    subpath = subpaths[inum][0]
                    ori_size = (ori_sizes[inum][1].item(),
                                ori_sizes[inum][0].item())

                    transform = trans.Compose([
                        trans.ToPILImage(),
                        trans.Scale(ori_size)
                    ])
                    outputImage = transform(pre)
                    filename = subpath.split('/')[1]
                    outputImage.save(os.path.join(save_final_path, filename))
                

if __name__ == '__main__':

    cudnn.benchmark = True

    start = time.time()

    model = CoS_Det_Net(cfg)
    model.cuda()
    print('Model has constructed!')

    model.load_state_dict(torch.load("results/cosal-joint-train@DUTS_class/checkpoint_2022_03_14-08_16_48/checkpoint_epoch16.pth")["state_dict"])
    # print('Model loaded from {}'.format(test_model_dir))

    test_net(model, 1)
    model.eval()
    print('total time {}'.format(time.time()-start))
