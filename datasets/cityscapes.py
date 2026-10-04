# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        # Submission copy: portable paths and no CUDA work in data workers.
        with open(os.path.join(root, list_path), encoding='utf-8') as split_file:
            self.img_list = [line.strip().split() for line in split_file if line.strip()]

        self.files = self.read_files()

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507])
        
        self.bd_dilate_size = bd_dilate_size

        # 预构建正向 LUT，避免 __getitem__ 中重复创建
        self._label_lut = np.full(256, ignore_label, dtype=np.uint8)
        for k, v in self.label_mapping.items():
            if 0 <= k < 256:
                self._label_lut[k] = v
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        if not inverse:
            # 正向：用预构建 LUT 一次完成映射，O(N) 且无循环开销
            return self._label_lut[label]
        # 反向（仅推理可视化时使用，频率低，保留原实现）
        temp = label.copy()
        for v, k in self.label_mapping.items():
            label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
                           cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(os.path.join(self.root, 'cityscapes', item['img']))
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(os.path.join(self.root, 'cityscapes', item['label']))
        label = self.convert_label(label)

        # 尝试加载预存的 edge 文件（与 label 同目录，名称加 _edge.png 后缀）
        label_full_path = os.path.join(self.root, 'cityscapes', item["label"])
        label_stem = os.path.splitext(label_full_path)[0]   # 去掉 .png/.bin 后缀
        edge_cache_path = label_stem + '_edge.png'
        if os.path.exists(edge_cache_path):
            precomputed_edge = cv2.imread(edge_cache_path, cv2.IMREAD_GRAYSCALE)
        else:
            precomputed_edge = None   # 回退到在线 Canny 计算

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size,
                                precomputed_edge=precomputed_edge)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
