import torch
import torch.nn as nn
import torchvision
import time
import matplotlib
import matplotlib.pyplot as plt
import torchvision

from kernels import Kernel
from shapes import Design

from simulate_image import simulate_image
from opc import OPC
from constant import MAX_DOSE, MIN_DOSE, NOMINAL_DOSE
from constant import MASKRELAX_SIGMOID_STEEPNESS
from constant import device
from constant import LITHOSIM_OFFSET, MASK_TILE_END_X, MASK_TILE_END_Y
from utils import write_image_file, unit_test_params


class GradientBlock(nn.Module):
    def __init__(self, kernels):
        super(GradientBlock, self).__init__()
        # init mask
        self.update_mask = nn.Sigmoid()
        self.m_mask = None
        self.filter = torch.zeros([2048, 2048])
        self.filter[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1
        # prepare for calculate pvband
        # image[0]: MAX_DOSE, FOCUS
        # image[1]: MIN_DOSE, DEFOCUS
        # image[2]: NOMINAL_DOSE, FOCUS
        self.m_image = [None, None, None]
        self.kernels = kernels

    def forward(self, params):
        self.m_mask = self.update_mask(MASKRELAX_SIGMOID_STEEPNESS * params) * self.filter
        # unit_test_kernel(self.m_mask, "/Users/zhubinwu/research/opc-hsd/cuilt/build/init_mask.txt", 2048, 2048)
        self.m_image[0] = simulate_image(self.m_mask, self.kernels["focus"].kernels,
                                       self.kernels["focus"].scales, MAX_DOSE, 15)
        self.m_image[1] = simulate_image(self.m_mask, self.kernels["defocus"].kernels,
                                       self.kernels["defocus"].scales, MIN_DOSE, 15)
        self.m_image[2] = simulate_image(self.m_mask, self.kernels["focus"].kernels,
                                         self.kernels["focus"].scales, NOMINAL_DOSE, 15)


if __name__ == "__main__":
    # kernel setting
    matplotlib.use('TkAgg')
    defocus_flag = 1
    conjuncture_flag = 1
    combo_flag = 1
    opt_kernels = {"focus": Kernel(35, 35, device), "defocus": Kernel(35, 35, device, defocus=defocus_flag),
                   "CT focus": Kernel(35, 35, device, conjuncture=conjuncture_flag),
                   "CT defocus": Kernel(35, 35, device, defocus=defocus_flag, conjuncture=conjuncture_flag)}

    # init design file and init params
    test_design = Design("../benchmarks/M1_test1" + ".glp")
    test_opc = OPC(test_design, hammer=1, sraf=0)
    test_opc.run()

    gradient = GradientBlock(opt_kernels)
    gradient(test_opc.m_params)
    # plt.imshow(write_image_file(gradient.m_image[0], MAX_DOSE))
    # plt.savefig("outer_image_iter1.png")
    plt.imshow(write_image_file(gradient.m_image[1], MIN_DOSE))
    plt.savefig("inner_image_iter1.png")
    # plt.imshow(write_image_file(gradient.m_image[2], NOMINAL_DOSE))
    # plt.savefig("nominal_image_iter1.png")
