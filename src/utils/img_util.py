import matplotlib.pyplot as plt
import cv2 as cv
import torch
from torch import Tensor
import numpy as np


def show_img_bbox(img, bboxs, filename="tmp.jpg"):
    if isinstance(bboxs, torch.Tensor):
        bboxs = bboxs.cpu().numpy()
    img *= 255
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for box in bboxs:
        box = list(map(int,box))
        cv.rectangle(img,
                     (box[0], box[1]),
                     (box[2], box[3]),
                     (0, 0, 255), 3)

    cv.imwrite(filename, img)
