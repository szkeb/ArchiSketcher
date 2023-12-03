from pathlib import Path
from enum import IntEnum
import os
import shutil

import numpy as np


def delete_all_files_from(dir_name: Path, silent=False):
    for file_name in os.listdir(dir_name):
        file = dir_name / file_name
        if os.path.isfile(file):
            if not silent:
                print('removing: ', file)
            os.remove(file)


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_file(src, dst):
    shutil.copyfile(src, dst)


def padding_length(collection):
    if type(collection) is int:
        return collection
    return len(str(len(collection)))


def pad_string(string: str, collection):
    return string.zfill(padding_length(collection))