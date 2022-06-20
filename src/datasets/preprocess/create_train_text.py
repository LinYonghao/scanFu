from utils.path import get_path, path2filename
import glob, os

"""
将某文件夹里面的图片作文训练集生成txt文件
"""


config = {
    # 图片文件目录
    "img_dir": get_path("D:\datasets\Fu\JPEGImages/"),
    "root_dir":get_path("D:\datasets\Fu/")
}


def main():
    glob_result = glob.glob(config["img_dir"] + "*")
    f = open(f"{config['root_dir']}train.txt","w")
    for idx,file in enumerate(glob_result):
        f.write(path2filename(file) + "\n")

    f.close()



if __name__ == '__main__':
    main()