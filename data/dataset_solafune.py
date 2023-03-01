import os
import random

import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from utils import utils_blindsr as blindsr


class DatasetSolafune(data.Dataset):
    """
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt["n_channels"] if opt["n_channels"] else 3
        self.paths_H = util.get_image_paths(opt["dataroot_H"])
        self.paths_L = util.get_image_paths(opt["dataroot_L"])
        print(f"H: {len(self.paths_H)}, L: {len(self.paths_L)}")

        assert self.paths_H, "Error: H path is empty."

    def __getitem__(self, index):
        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)
        img_L_name, _ = os.path.splitext(os.path.basename(L_path))
        H, W, C = img_L.shape

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H_name, _ = os.path.splitext(os.path.basename(H_path))
        H, W, C = img_H.shape

        assert img_H_name[:-4] == img_L_name[:-3]

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        return {"L": img_L, "H": img_H, "L_path": L_path, "H_path": H_path}

    def __len__(self):
        return len(self.paths_H)
