import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
import time
from parameter import *
from dataset import get_loader
import math
from parameter import *
import parameter
import datetime
from model import *

def test_net(model, batch_size):

    for dataset_idx in range(len(test_dir_img)):
        dataset_name = test_dir_img[dataset_idx].split('/')[-2]

        print('testing {}'.format(test_dir_img[dataset_idx]))

        test_loader = get_loader(test_dir_img[dataset_idx], img_size, 1, gt_root=None, mode='test', num_thread=1)
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
            group_subpaths = data_batch[1]
            group_ori_sizes = data_batch[2]

            # print(cos_imgs_groups.shape)
            # print(len(group_subpaths),group_subpaths[0])
            num = cos_imgs_groups.shape[0]
            group_nums = num//max_num + (1 if num%max_num>0 else 0)

            batch_groups_paths = []
            batch_imgs = []
            batch_group_ori_sizes = []
            for i in range(group_nums-1):
                batch_groups_paths.append(group_subpaths[i*max_num:(i+1)*max_num])
                batch_imgs.append(cos_imgs_groups[i*max_num:(i+1)*max_num])
                batch_group_ori_sizes.append(group_ori_sizes[i*max_num:(i+1)*max_num])

            batch_groups_paths.append(group_subpaths[-max_num:])
            batch_imgs.append(cos_imgs_groups[-max_num:])
            batch_group_ori_sizes.append(group_ori_sizes[-max_num:])

            for group, subpaths,ori_sizes in zip(batch_imgs, batch_groups_paths,batch_group_ori_sizes):
                # print(group.shape)
                # print(len(subpaths))
                inputs = []
                for img in group:
                    inputs.append({"image": img, "height": img_size, "width": img_size})

                nms_boxes,pred_vector,output_binary = model(inputs)  
                output_binary = torch.round(torch.sigmoid(output_binary))
                pos_imgs_boxes = boxes_preded(nms_boxes,pred_vector)

                output_binary = binary_after_boxes(output_binary, pos_imgs_boxes)
                output = output_binary
                # print(output)
                saved_root = save_test_path_root + dataset_name
                # save_final_path = saved_root + '/CADC_' + test_model + '/' + subpaths[0][0].split('/')[0] + '/'
                save_final_path = saved_root + '/CADC/' + subpaths[0][0].split('/')[0] + '/'
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

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    cudnn.benchmark = True

    start = time.time()

    model = CoS_Det_Net()
    model.cuda()
    print('Model has constructed!')

    model.load_state_dict(torch.load(test_model_dir))
    print('Model loaded from {}'.format(test_model_dir))

    test_net(model, 1)
    print('total time {}'.format(time.time()-start))
