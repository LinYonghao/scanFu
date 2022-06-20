def windows_path2wsl_path(window_path: str):
    """

    :param window_path:
    :return:
    :example: D:\datasets\Fu   convert-> /mnt/d/datasets/Fu
    """
    disk_flag_position = window_path.find(":")
    disk_flag = window_path[disk_flag_position - 1:disk_flag_position]
    return "/mnt/" + disk_flag.lower() + "/" + window_path[disk_flag_position + 2:].replace("\\", "/")
