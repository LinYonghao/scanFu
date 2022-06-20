from utils.path import get_path
import glob, os

"""
将某文件夹的图片转换成从0到N的图片
"""


config = {
    # 图片文件目录
    "img_dir": get_path("D:\datasets\Fu\JPEGImages/")
}


def main():
    glob_result = glob.glob(config["img_dir"] + "*")
    for idx,file in enumerate(glob_result):
        os.rename(file,f"{config['img_dir']}{idx}.jpg")


if __name__ == '__main__':
    main()