import torch

from constant import device
from constant import KERNEL_X, KERNEL_Y
from constant import OPC_TILE_X, OPC_TILE_Y
from constant import TARGET_INTENSITY

class Simulator(object):
    def __init__(self):
        self.imx = OPC_TILE_X
        self.imy = OPC_TILE_Y
        self.kernel_x = KERNEL_X
        self.kernel_y = KERNEL_Y

    def shift(self, cmask):
        shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex128, device=device)
        shift_cmask[:, :self.imx // 2, :self.imy // 2] = cmask[:, self.imx // 2:, self.imy // 2:]  # 1 = 4
        shift_cmask[:, :self.imx // 2, self.imy // 2:] = cmask[:, self.imx // 2:, :self.imy // 2]  # 2 = 3
        shift_cmask[:, self.imx // 2:, :self.imy // 2] = cmask[:, :self.imx // 2, self.imy // 2:]  # 3 = 2
        shift_cmask[:, self.imx // 2:, self.imy // 2:] = cmask[:, :self.imx // 2, :self.imy // 2]  # 4 = 1
        return shift_cmask

    def kernel_mult(self, knx, kny, kernel, mask_fft, kernel_num):
        imxh = self.imx // 2
        imxy = self.imy // 2
        xoff = imxh - knx // 2
        yoff = imxy - kny // 2
        kernel = kernel.permute(2, 0, 1)
        temp_mask_fft = mask_fft[0, xoff:xoff + knx, yoff:yoff + kny]
        output = torch.zeros([kernel_num, self.imx, self.imy], dtype=torch.complex128, device=device)
        output[:, xoff:xoff + knx, yoff:yoff + kny] = temp_mask_fft * kernel[:kernel_num, :, :]
        return output

    def compute_image(self, cmask, kernel, scale, workx, worky, dose, kernel_level):
        kernel_num = kernel_level
        cmask = torch.unsqueeze(cmask, 0)
        cmask = self.shift(cmask)
        cmask_fft = torch.fft.fft2(cmask, norm="forward")
        cmask_fft = self.shift(cmask_fft)
        temp = self.shift(self.kernel_mult(self.kernel_x, self.kernel_y, kernel, cmask_fft, kernel_num))
        temp = self.shift(torch.fft.ifft2(temp, norm="forward"))
        if kernel_level == 1:
            return temp[0]
        elif kernel_level == 15:
            scale = scale[:kernel_num].unsqueeze(1).unsqueeze(2)
            mul_fft = torch.sum(scale * torch.pow(torch.abs(temp), 2), dim=0).to(device)
            return mul_fft

    def mask_float(self, mask, dose):
        return (dose * mask).to(torch.complex128)

    def simulate_image(self, mask, kernel, scale, dose, kernel_level):
        cmask = self.mask_float(mask, dose)
        image = self.compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)
        return image

    def convolve_image(self, kernel, scale, dose, kernel_level, mask=None, matrix=None):
        if mask is not None:
            return self.simulate_image(mask, kernel, scale, dose, kernel_level)
        elif matrix is not None:
            cmask = matrix
            image = self.compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)
            return image

    def calculate_pvband(self, inner, outer):
        pvb_area = torch.sum(torch.logical_and(outer >= TARGET_INTENSITY, inner < TARGET_INTENSITY))
        return pvb_area

if __name__ == "__main__":
    simulator = Simulator()
    mask = torch.randn([2048, 2048])
    kernel = torch.randn([35, 35, 1])
    scales = torch.randn([1])
    dose = 1.02
    kernel_level = 1
    simulator.simulate_image(mask, kernel, scales, dose, kernel_level)