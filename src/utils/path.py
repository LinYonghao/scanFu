import platform
from utils.wsl2 import windows_path2wsl_path


def get_path(path, is_wsl=True):
    # 'Linux', 'Windows' or 'Java'.

    if is_wsl is False:
        return path

    if platform.system() == "Windows":
        return path
    else:
        return windows_path2wsl_path(path)


def path2filename(path: str):
    flag = path.rfind("\\")
    if flag == -1:
        flag = path.rfind("/")

    return path[flag + 1:]


if __name__ == '__main__':
    print(path2filename("c:\\abc\\cbd.jpg"))
