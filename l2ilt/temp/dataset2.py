import torch
import torchvision.transforms as trans
import cv2
import os
from constant import *

class ICCADTrain2(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.hammer_list = sorted(os.listdir(root_dir + "train/hammer/"))
        self.target_list = sorted(os.listdir(root_dir + "train/target/"))

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        hammer = trans.ToTensor()(cv2.imread(self.root_dir + "train/hammer/" + self.hammer_list[index]))
        target = trans.ToTensor()(cv2.imread(self.root_dir + "train/target/" + self.target_list[index]))
        hammer = hammer[0].float()
        param = torch.zeros(hammer.size(), dtype=torch.float32)
        temp_param = torch.ones([MASK_TILE_END_Y - LITHOSIM_OFFSET,
                                  MASK_TILE_END_X - LITHOSIM_OFFSET], dtype=torch.float32)
        temp_param[hammer[LITHOSIM_OFFSET:MASK_TILE_END_Y,
                    LITHOSIM_OFFSET:MASK_TILE_END_X] < MASK_PRINTABLE_THRESHOLD] = -1
        param[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = temp_param
        target = target[0].float()
        return {"param": param, "target": target}

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.hammer_list)

if __name__ == "__main__":
    mydataset = ICCADTrain("ICCAD2013/")
    print(mydataset.hammer_list)
    print(mydataset.target_list)