import torch
import torchvision.transforms as trans
import cv2
import os
from constant import *
# from dataset2 import ICCADTrain2

class ICCADTrain(torch.utils.data.Dataset):
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
        hammer = hammer[0:1, :, :].float()
        # print(hammer.size())
        param = torch.zeros(hammer.size(), dtype=torch.float32)
        # print(param.size())
        temp_param = torch.ones([1, MASK_TILE_END_Y - LITHOSIM_OFFSET,
                                  MASK_TILE_END_X - LITHOSIM_OFFSET], dtype=torch.float32)
        # print(temp_param.size())
        # print(hammer[0, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] < MASK_PRINTABLE_THRESHOLD)
        temp_param[hammer[0:1, LITHOSIM_OFFSET:MASK_TILE_END_Y,
                    LITHOSIM_OFFSET:MASK_TILE_END_X] < MASK_PRINTABLE_THRESHOLD] = -1
        param[0, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = temp_param
        target = target[0:1, :, :].float()
        return {"param": param, "target": target}

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.hammer_list)

if __name__ == "__main__":
    mydataset = ICCADTrain("ICCAD2013/")
    # mydataset_gt = ICCADTrain2("ICCAD2013/")
    # data_pair = mydataset[0]
    # data_pair_gt = mydataset_gt[0]
    # param = data_pair["param"]
    # param_gt = data_pair_gt["param"]
    # target = data_pair["target"]
    # target_gt = data_pair_gt["target"]
    # print(param_gt.equal(param[0]))
    # print(target_gt.equal(target[0]))