import os.path

import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

from utils.path import get_path
import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv

import albumentations as A


class FuDataset(Dataset):
    '''
        修改记录
        v1.0 没有做数据增加 用了原图进行训练和验证
        v2.0 加了数据增加（图片翻转）
        返回 image,target,item 三个参数都是torch.Tensor类型 image 做了 归一化处理
            图片  ,标签数据,图片文件id
    '''

    def __init__(self, img_dir, xml_dir, txt, transforms=None):
        """
        :param root_dir: 数据根目录d
        :param img_dir: 图片根目录
        :param txt: 数据文本文件 分隔符为换行符\n
        """
        self.file_list = []
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transforms
        with open(txt) as f:
            text = f.read()
            text_splited = text.split("\n")
            for item in text_splited:
                self.file_list.append(item)

    def __getitem__(self, item):
        current_filename = self.file_list[item]
        image = cv.imread(os.path.join(self.img_dir, current_filename))
        print(current_filename)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # 获取xml
        file_without_ext = current_filename[:current_filename.find(".")]
        anno = ET.parse(
            os.path.join(self.xml_dir, file_without_ext + '.xml'))
        bboxs = []
        labels = []
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            bboxs.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            labels.append(VOC_BBOX_LABEL_NAMES.index(name))

        target = {}
        target['boxes'] = torch.tensor(bboxs)
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([item])


        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels,
            }

            sample = self.transform(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        else:
            image = ToTensor()(image)

        return image, target, item

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.Flip(0.5),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    @staticmethod
    def get_valid_transform():
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


VOC_BBOX_LABEL_NAMES = (
    'background',
    'fu',
)

if __name__ == '__main__':
    iters = iter(FuDataset(
        img_dir=get_path("D:\datasets\Fu\JPEGImages/"),
        txt=get_path("D:\datasets\Fu\\train.txt"),
        xml_dir=get_path("D:\datasets\Fu\Annotaions/")
    ))

    next(iters)
