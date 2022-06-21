import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np
import cv2 as cv
from torchvision.transforms import ToTensor

from datasets.FuDataset import FuDataset
from model.core.fasterrcnnn import fasterrcnn_resnet50_fpn
from utils.img_util import show_img_bbox
from utils.path import get_path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 函数式编程 拿模型
model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
# 两分类任务 0 backgroud 1 fu
num_classes = 2
# 获取ROI检测投输入特征
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 替换检测头
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# 加载权重
model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn.pth", map_location=device))
model.eval()

_ = model.to(device)

score_threshold = 0.7
image_outputs = []

test_dataset = FuDataset(
        img_dir=get_path("D:\datasets\Fu\JPEGImages/"),
        txt=get_path("D:\datasets\Fu\\test.txt"),
        xml_dir=get_path("D:\datasets\Fu\Annotaions/")
    )
def collate_fn(batch):
  return tuple(zip(*batch))

test_data_loader = DataLoader(
  test_dataset,
  batch_size=1,
  shuffle=False,
  num_workers=4,
  collate_fn=collate_fn
)

for images, target,image_ids in test_data_loader:
  images = list(image.to(device) for image in images)
  outputs = model(images)

  for image_id, output in zip(image_ids, outputs):
    boxes = output['boxes'].data.cpu().numpy()
    scores = output['scores'].data.cpu().numpy()

    mask = scores >= score_threshold
    boxes = boxes[mask].astype(np.int32)
    scores = scores[mask]
    print(images[0].cpu().numpy().shape)
    show_img_bbox(images[0].cpu().permute(1, 2, 0).numpy(),boxes,f"./{image_id}.jpg")
    image_outputs.append((image_id, boxes, scores))
