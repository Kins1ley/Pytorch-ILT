import torch

from constant import device
from constant import KERNEL_X, KERNEL_Y
from constant import OPC_TILE_X, OPC_TILE_Y


class Simulator(object):
    def __init__(self):
        self.imx = OPC_TILE_X
        self.imy = OPC_TILE_Y
        self.kernel_x = KERNEL_X
        self.kernel_y = KERNEL_Y

    def shift(self, cmask):
        shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex128, device=device)
        shift_cmask[:self.imx // 2, :self.imy // 2] = cmask[self.imx // 2:, self.imy // 2:]  # 1 = 4
        shift_cmask[:self.imx // 2, self.imy // 2:] = cmask[self.imx // 2:, :self.imy // 2]  # 2 = 3
        shift_cmask[self.imx // 2:, :self.imy // 2] = cmask[:self.imx // 2, self.imy // 2:]  # 3 = 2
        shift_cmask[self.imx // 2:, self.imy // 2:] = cmask[:self.imx // 2, :self.imy // 2]  # 4 = 1
        return shift_cmask

    def kernel_mult(self, kernel, mask_fft):
        imxh = self.imx // 2
        imxy = self.imy // 2
        xoff = imxh - self.kernel_x // 2
        yoff = imxy - self.kernel_y // 2
        output = torch.zeros(mask_fft.size(), dtype=torch.complex128, device=device)
        output[xoff:xoff + self.kernel_x, yoff:yoff + self.kernel_y] = \
            torch.mul(kernel, mask_fft[xoff:xoff + self.kernel_x, yoff:yoff + self.kernel_y])
        return output

    def compute_image(self, cmask, kernel, scale, workx, worky, dose, kernel_level):
        kernel_num = kernel_level
        cmask = self.shift(cmask)
        cmask_fft = torch.fft.fft2(cmask)
        cmask_fft = self.shift(cmask_fft)
        mul_fft = torch.zeros(cmask_fft.size(), dtype=torch.float64, device=device)
        for i in range(kernel_num):
            temp = self.shift(self.kernel_mult(kernel[:, :, i], cmask_fft))
            temp = self.shift(torch.fft.ifft2(temp))
            mul_fft += scale[i] * torch.pow(torch.abs(temp), 2)
        return mul_fft

    def mask_float(self, mask, dose):
        return (dose * mask).to(torch.complex128)

    def simulate_image(self, mask, kernel, scale, dose, kernel_level):
        cmask = self.mask_float(mask, dose)
        image = self.compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)
        return image