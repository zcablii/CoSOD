from torch.utils import data
import torch
import matplotlib.pyplot as plt

import dataset.transforms as trans
from PIL import Image
import random
import os

torch.multiprocessing.set_sharing_strategy('file_system')
class TrainDataset(data.Dataset):
    def __init__(self, cfg):
        # Path Pool
        self.root = cfg.DATA.ROOT
        self.datasets = {}
        self.max_class_dataset = ['', 0]
        for name, img_root, gt_root in zip(cfg.DATA.NAMES,
                                           cfg.DATA.IMAGE_DIRS,
                                           cfg.DATA.GT_DIRS):
            img_root = os.path.join(self.root, img_root)
            gt_root = os.path.join(self.root, gt_root)
            classes = os.listdir(img_root)
            # [img_root+dir1, ..., img_root+dir2]
            img_dir_paths = list(map(lambda x: os.path.join(img_root, x), classes))
            gt_dir_paths = list(map(lambda x: os.path.join(gt_root, x), classes))

            # [[name00,..., 0N],..., [M0,..., MN]]
            img_name_list = [os.listdir(i_dir) for i_dir in img_dir_paths]
            gt_name_list = [map(lambda x: x[:-3] + 'png', iname_list) for iname_list in img_name_list]

            # [[path00,..., 0N],..., [M0,..., MN]]
            self.img_path_list = [
                list(map(lambda x: os.path.join(img_dir_paths[idx], x),
                         img_name_list[idx]))
                for idx in range(len(img_dir_paths))
            ]
            self.gt_path_list = [
                list(map(lambda x: os.path.join(gt_dir_paths[idx], x),
                         gt_name_list[idx]))
                for idx in range(len(gt_dir_paths))
            ]
            self.datasets[name] = (self.img_path_list,
                                   self.gt_path_list,
                                   len(classes))
            if len(classes) >= self.max_class_dataset[1]:
                self.max_class_dataset[0] = name
                self.max_class_dataset[1] = len(classes)

        # Transform
        self.transform = trans.Compose([
            trans.ToTensor_BGR(),
        ])
        self.t_transform = trans.Compose([
            trans.ToTensor(),
        ])
        self.label_32_transform = trans.Compose([
            trans.Scale((cfg.DATA.IMAGE_H // 8, cfg.DATA.IMAGE_W // 8),
                        interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])
        self.label_64_transform = trans.Compose([
            trans.Scale((cfg.DATA.IMAGE_H // 4, cfg.DATA.IMAGE_W // 4),
                        interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])
        self.label_128_transform = trans.Compose([
            trans.Scale((cfg.DATA.IMAGE_H // 2, cfg.DATA.IMAGE_W // 2),
                        interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])

        # Other HyperParameters
        self.size = (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W)
        self.scale_size = (cfg.DATA.SCALE_H, cfg.DATA.SCALE_W)
        self.cat_size = cfg.DATA.IMAGE_H + cfg.DATA.IMAGE_W
        self.group_num = cfg.DATA.MAX_NUM
        self.syns = list(zip(cfg.DATA.SYN_NAMES,
                             cfg.DATA.SYN_TYPES,
                             cfg.DATA.SYN_IMAGE_DIRS,
                             cfg.DATA.SYN_GT_DIRS))

    def __getitem__(self, item):
        sample_dataset_name = random.choice(list(self.datasets.keys()))
        if sample_dataset_name == self.max_class_dataset[0]:
            index = item
        else:
            index = item % (self.datasets[sample_dataset_name][2] - 1)
        sample_dataset = self.datasets[sample_dataset_name]
        img_paths = sample_dataset[0][index]
        gt_paths = sample_dataset[1][index]

        now_group_num = len(img_paths)
        if now_group_num > self.group_num:
            sampled_list = random.sample(range(now_group_num), self.group_num)
            new_img_paths = [img_paths[i] for i in sampled_list]
            img_paths = new_img_paths
            new_gt_paths = [gt_paths[i] for i in sampled_list]
            gt_paths = new_gt_paths
            now_group_num = self.group_num

        imgs = torch.Tensor(now_group_num, 3, self.size[0], self.size[1])
        gts = torch.Tensor(now_group_num, 1, self.size[0], self.size[1])
        gts_32 = torch.Tensor(now_group_num, 1, self.size[0] // 8, self.size[1] // 8)
        gts_64 = torch.Tensor(now_group_num, 1, self.size[0] // 4, self.size[1] // 4)
        gts_128 = torch.Tensor(now_group_num, 1, self.size[0] // 2, self.size[1] // 2)
        origin_sizes = []
        subpaths = []
        for syn_name, syn_types, syn_img_root, syn_gt_root in self.syns:
            for idx in range(now_group_num):
                if sample_dataset_name == syn_name:
                    syn_types += ["origin"]
                    strategy = random.choice(syn_types)
                    if False: #strategy == "origin":
                        img_path = img_paths[idx]
                        gt_path = gt_paths[idx]
                    else:
                        strategy = syn_types[0]
                        select_num = random.randint(1, strategy[2])
                        dir_ = img_paths[idx].split('/')[-2]
                        name = img_paths[idx].split('/')[-1][:-4] + strategy[1] + "{}.png".format(select_num)
                        syn_img_root_new = os.path.join(self.root, syn_img_root.format(strategy[0], dir_))
                        img_path = os.path.join(syn_img_root_new, name)
                        if strategy[3]:
                            gt_path = gt_paths[idx]
                        else:
                            syn_gt_root_new = os.path.join(self.root, syn_gt_root.format(strategy[0], dir_))
                            gt_path = os.path.join(syn_gt_root_new, name)
                else:
                    img_path = img_paths[idx]
                    gt_path = gt_paths[idx]

                img = Image.open(img_path).convert('RGB')
                gt = Image.open(gt_path).convert('L')
                subpaths.append(
                    os.path.join(img_paths[idx].split('/')[-2], img_paths[idx].split('/')[-1][:-4] + '.png'))
                origin_sizes.append((img.size[0], img.size[1]))
                # random crop
                random_size = self.scale_size
                new_img = trans.Scale(random_size)(img)
                new_gt = trans.Scale(random_size, interpolation=Image.NEAREST)(gt)

                h, w = new_img.size
                if h != self.size[0] and w != self.size[1]:
                    x1 = random.randint(0, w - self.size[0])
                    y1 = random.randint(0, h - self.size[1])
                    new_img = new_img.crop((x1, y1, x1 + self.size[0], y1 + self.size[1]))
                    new_gt = new_gt.crop((x1, y1, x1 + self.size[0], y1 + self.size[1]))

                # random flip
                if random.random() < 0.5:
                    new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_gt = new_gt.transpose(Image.FLIP_LEFT_RIGHT)

                new_img = self.transform(new_img)
                gt_256 = self.t_transform(new_gt)
                gt_32 = self.label_32_transform(new_gt)
                gt_64 = self.label_64_transform(new_gt)
                gt_128 = self.label_128_transform(new_gt)

                imgs[idx] = new_img
                gts[idx] = gt_256
                gts_128[idx] = gt_128
                gts_64[idx] = gt_64
                gts_32[idx] = gt_32
        return imgs, gts, gts_128, gts_64, gts_32, subpaths, origin_sizes

    def __len__(self):
        return self.max_class_dataset[1]


class TestDataset(data.Dataset):
    def __init__(self, cfg, root, img_root, gt_root):
        self.root = root
        self.img_root = os.path.join(self.root, img_root)
        self.gt_root = os.path.join(self.root, gt_root)
        self.classes = os.listdir(self.img_root)
        self.img_dirs = list(map(lambda x: os.path.join(self.img_root, x), self.classes))
        self.gt_dirs = list(map(lambda x: os.path.join(self.gt_root, x), self.classes))

        self.size = (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W)
        self.transform = trans.Compose([
            trans.Scale((cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W)),
            trans.ToTensor_BGR(),
        ])
        self.t_transform = trans.Compose([
            trans.Scale((cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W), interpolation=Image.NEAREST),
            trans.ToTensor(),
        ])

    def __getitem__(self, item):
        img_names = os.listdir(self.img_dirs[item])
        gt_names = [iname_list[:-3] + 'png' for iname_list in img_names]
        num = len(img_names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), img_names))
        gt_paths = list(
            map(lambda x: os.path.join(self.gt_dirs[item], x), gt_names))

        imgs = torch.Tensor(num, 3, self.size[0], self.size[1])
        gts = torch.Tensor(num, 1, self.size[0], self.size[1])

        sub_paths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')
            sub_paths.append(os.path.join(img_paths[idx].split('/')[-2], img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img
            gt = self.t_transform(gt)
            gts[idx] = gt
        return imgs, gts, sub_paths, ori_sizes

    def __len__(self):
        return len(self.classes)


def get_loader(cfg, mode='train'):
    if mode == 'train':
        dataset = TrainDataset(cfg)
        return data.DataLoader(dataset=dataset,
                               batch_size=cfg.DATA.BATCH_SIZE,
                               shuffle=True,
                               num_workers=cfg.DATA.WORKERS,
                               pin_memory=cfg.DATA.PIN)

    else:
        if mode == "eval":
            data_loaders = []
            for test_image_dir, test_gt_dir in zip(cfg.VAL.IMAGE_DIRS, cfg.VAL.GT_DIRS):
                dataset = TestDataset(cfg,
                                      cfg.VAL.ROOT,
                                      test_image_dir,
                                      test_gt_dir)
                data_loaders.append(data.DataLoader(dataset=dataset,
                                                    batch_size=cfg.DATA.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=cfg.DATA.WORKERS,
                                                    pin_memory=cfg.DATA.PIN))
            return data_loaders
        else:
            data_loaders = []
            for test_image_dir, test_gt_dir in zip(cfg.TEST.IMAGE_DIRS, cfg.TEST.GT_DIRS):
                dataset = TestDataset(cfg,
                                      cfg.TEST.ROOT,
                                      test_image_dir,
                                      test_gt_dir)
                data_loaders.append(data.DataLoader(dataset=dataset,
                                                    batch_size=cfg.DATA.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=cfg.DATA.WORKERS,
                                                    pin_memory=cfg.DATA.PIN))
            return data_loaders
