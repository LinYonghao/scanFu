import matplotlib.pyplot as plt
import cv2 as cv
from torch import Tensor
import numpy as np


def show_img_bbox(img,bboxs,filename="tmp.jpg"):
    img *= 255
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    for box in bboxs:
        cv.rectangle(img,
                     (box[0], box[1]),
                     (box[2], box[3]),
                     (0, 0, 255), 3)

    cv.imwrite(filename,img)
