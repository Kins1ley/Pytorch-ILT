import torch
import sys
import time
import torch.nn as nn
sys.path.append("..")
from constant import *
from dataset import ICCADTrain
from kernels import Kernel
from utils import write_image_file
import matplotlib.pyplot as plt

class Simulator(nn.Module):
    def __init__(self):
        super(Simulator, self).__init__()
        self.imx = OPC_TILE_X
        self.imy = OPC_TILE_Y
        self.kernel_x = KERNEL_X
        self.kernel_y = KERNEL_Y

    def shift(self, cmask):
        shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex64, device=device)
        shift_cmask[:, :, :self.imx // 2, :self.imy // 2] = cmask[:, :, self.imx // 2:, self.imy // 2:]  # 1 = 4
        shift_cmask[:, :, :self.imx // 2, self.imy // 2:] = cmask[:, :, self.imx // 2:, :self.imy // 2]  # 2 = 3
        shift_cmask[:, :, self.imx // 2:, :self.imy // 2] = cmask[:, :, :self.imx // 2, self.imy // 2:]  # 3 = 2
        shift_cmask[:, :, self.imx // 2:, self.imy // 2:] = cmask[:, :, :self.imx // 2, :self.imy // 2]  # 4 = 1
        return shift_cmask

    def kernel_mult(self, knx, kny, kernel, mask_fft, kernel_num):
        imxh = self.imx // 2
        imxy = self.imy // 2
        xoff = imxh - knx // 2
        yoff = imxy - kny // 2
        kernel = kernel.permute(2, 0, 1)
        temp_mask_fft = mask_fft[:, :, xoff:xoff + knx, yoff:yoff + kny]
        output = torch.zeros([1, kernel_num, self.imx, self.imy], dtype=torch.complex64, device=device)
        output[0, :, xoff:xoff + knx, yoff:yoff + kny] = temp_mask_fft * kernel[:kernel_num, :, :]
        return output

    def compute_image(self, cmask, kernel, scale, workx, worky, dose, kernel_level):
        kernel_num = kernel_level
        # cmask = torch.unsqueeze(cmask, 0)
        cmask = self.shift(cmask)
        cmask_fft = torch.fft.fft2(cmask, norm="forward")
        cmask_fft = self.shift(cmask_fft)
        temp = self.shift(self.kernel_mult(self.kernel_x, self.kernel_y, kernel, cmask_fft, kernel_num))
        temp = self.shift(torch.fft.ifft2(temp, norm="forward"))
        if kernel_level == 1:
            return temp[0]
        elif kernel_level == 15 or kernel_level == 24:
            scale = scale[:kernel_num]
            # print(scale.size())
            mul_fft = torch.sum(scale * torch.pow(torch.abs(temp), 2), dim=1, keepdim=True).to(device)
            return mul_fft

    def mask_float(self, mask, dose):
        return (dose * mask).to(torch.complex64)

    def forward(self, mask, kernel, scale, dose, kernel_level):
        # mask [N, C=1, H, W], kernel [num_kernel, H, W]
        cmask = self.mask_float(mask, dose)
        image = self.compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)
        return image


class GradientBlock(nn.Module):
    def __init__(self):
        super(GradientBlock, self).__init__()
        self.theta_z = PHOTORISIST_SIGMOID_STEEPNESS
        self.target_intensity = TARGET_INTENSITY
        self.theta_m = MASKRELAX_SIGMOID_STEEPNESS
        self.gamma = 4
        # self.conv0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.update_mask = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.step_size = torch.ones([1, OPC_TILE_Y, OPC_TILE_X], dtype=torch.float32, device=device) * 0.5
        self.simulator = Simulator()
        self.filter = torch.zeros([1, OPC_TILE_Y, OPC_TILE_X], dtype=torch.float32).to(device)
        self.filter[0, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1.0