import os
from os.path import join


def list_items(fpath, ext):
    tr_sub_dir = [join(fpath, i) for i in os.listdir(fpath)]
    return [join(j, i) for j in tr_sub_dir for i in os.listdir(j) if i.endswith(ext)]

