import os.path

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

from utils.path import get_path
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv

class FuDataset(Dataset):

    def __init__(self, root_dir,img_dir,xml_dir, txt):
        """
        :param root_dir: 数据根目录
        :param img_dir: 图片根目录
        :param txt: 数据文本文件 分隔符为换行符\n
        """
        self.file_list = []
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        with open(txt) as f:
            text = f.read()
            text_splited = text.split("\n")
            for item in text_splited:
                self.file_list.append(item)





    def __getitem__(self, item):
        current_filename = self.file_list[item]
        image = cv.imread(os.path.join(self.img_dir,current_filename))
        print(current_filename)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # 获取xml
        file_without_ext = current_filename[:current_filename.find(".")]

        anno = ET.parse(
            os.path.join(self.xml_dir,  file_without_ext + '.xml'))
        bboxs = []
        labels = []
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bboxs.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            labels.append(VOC_BBOX_LABEL_NAMES.index(name))


        target = {}
        target['boxes'] = torch.tensor(bboxs)
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([item])

        image = ToTensor()(image)
        return image,target,item

    def __len__(self):
        return len(self.file_list)

VOC_BBOX_LABEL_NAMES = (
    'background',
    'fu',
)



if __name__ == '__main__':

    iters = iter(FuDataset(
        root_dir=get_path("D:\datasets\Fu"),
        img_dir=get_path("D:\datasets\Fu\JPEGImages/"),
        txt=get_path("D:\datasets\Fu\\train.txt"),
        xml_dir=get_path("D:\datasets\Fu\Annotaions/")
    ))

    next(iters)