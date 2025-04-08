#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from glob import glob
from pathlib import Path
from errno import EEXIST
from os import makedirs, path
import os
import re

from natsort import natsorted

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if fname.split("_")[-1].isnumeric()]
    return max(saved_iters)

def get_last_ply_path(input_path: Path, return_all: bool = False):
    input_path = Path(input_path)
    ply_paths = glob(str(input_path / "point_cloud"/ "iteration_*" / "point_cloud.ply"))
    ply_paths = [p for p in ply_paths if re.match("^iteration_[0-9]*$", p.split("/")[-2]) is not None]
    ply_paths = natsorted(ply_paths)
    ply_path = ply_paths[-1]

    if return_all:
        return ply_paths
    else:
        return ply_path