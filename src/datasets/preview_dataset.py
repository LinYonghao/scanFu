from utils.img_util import show_img_bbox
from FuDataset import FuDataset
from utils.path import get_path

train_dataset = FuDataset(
    root_dir=get_path("D:\datasets\Fu"),
    img_dir=get_path("D:\datasets\Fu\JPEGImages/"),
    txt=get_path("D:\datasets\Fu\\train.txt"),
    xml_dir=get_path("D:\datasets\Fu\Annotaions/")
)

datas = [train_dataset[i] for i in range(2)]
imgs = [d[0].permute(1, 2, 0).numpy() for d in datas]
for (image, target, idx), img in zip(datas, imgs):
    show_img_bbox(img, target['boxes'])
