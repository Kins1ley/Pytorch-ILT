import torch
import torch.nn as nn
import torchvision
import time
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import sys
import platform
sys.path.append("..")

from kernels import Kernel
from shapes import Design

import simulator
import simulator2
from opc import OPC
from constant import MAX_DOSE, MIN_DOSE, NOMINAL_DOSE
from constant import MASKRELAX_SIGMOID_STEEPNESS
from constant import device
from constant import LITHOSIM_OFFSET, MASK_TILE_END_X, MASK_TILE_END_Y
from utils import write_image_file


class GradientBlock(nn.Module):
    def __init__(self, kernels):
        super(GradientBlock, self).__init__()
        # init mask
        self.update_mask = nn.Sigmoid()
        self.m_mask = None
        self.filter = torch.zeros([2048, 2048]).to(device)
        self.filter[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1
        # prepare for calculate pvband
        # image[0]: MAX_DOSE, FOCUS
        # image[1]: MIN_DOSE, DEFOCUS
        # image[2]: NOMINAL_DOSE, FOCUS
        self.m_image = [None, None, None]
        self.kernels = kernels
        self.simulator = simulator2.Simulator()
        # self.simulator2 = simulator2.Simulator()

    def forward(self, params):
        self.m_mask = self.update_mask(MASKRELAX_SIGMOID_STEEPNESS * params) * self.filter
        self.m_image[0] = self.simulator.simulate_image(self.m_mask, self.kernels["focus"].kernels,
                                       self.kernels["focus"].scales, MAX_DOSE, 15)
        self.m_image[1] = self.simulator.simulate_image(self.m_mask, self.kernels["defocus"].kernels,
                                       self.kernels["defocus"].scales, MIN_DOSE, 15)
        self.m_image[2] = self.simulator.simulate_image(self.m_mask, self.kernels["focus"].kernels,
                                         self.kernels["focus"].scales, NOMINAL_DOSE, 15)


def check_equal_image(cpp_file, py_image):
    '''
    check if the result of the python image is equal to the cpp iamge
    :param cpp_file: the position index of the pixels of cpp image that are not 0, represented as:
        1368 1048
        1368 1049
        1368 1050
        1368 1051
        1368 1052
    :param py_image: the tensor of one channel of the python output image
    :return: None
    '''
    with open(cpp_file) as f:
        cpp_outer_image = f.readlines()
    for i in range(len(cpp_outer_image)):
        cpp_outer_image[i] = cpp_outer_image[i].split(" ")
        cpp_outer_image[i][1] = cpp_outer_image[i][1][:-1]

    for i in range(len(cpp_outer_image)):
        for j in range(2):
            cpp_outer_image[i][0] = int(cpp_outer_image[i][0])
            cpp_outer_image[i][1] = int(cpp_outer_image[i][1])

    print(py_image[cpp_outer_image[0][0], cpp_outer_image[0][1]])
    for pos in cpp_outer_image:
        if py_image[pos[0], pos[1]] == 0:
            print("error")


if __name__ == "__main__":
    # kernel setting
    if platform.system() == "Darwin":
        matplotlib.use('TkAgg')
    defocus_flag = 1
    conjuncture_flag = 1
    combo_flag = 1
    opt_kernels = {"focus": Kernel(35, 35), "defocus": Kernel(35, 35, defocus=defocus_flag),
                   "CT focus": Kernel(35, 35, conjuncture=conjuncture_flag),
                   "CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag)}

    # init design file and init params

    start = time.time()
    test_design = Design("../benchmarks/M1_test1" + ".glp")
    # end = time.time()
    # print(end-start)
    # start = time.time()
    test_opc = OPC(test_design, hammer=1, sraf=0)
    # end = time.time()
    # print(end-start)
    # start = time.time()
    test_opc.run()
    # end = time.time()
    # print(end-start)
    gradient = GradientBlock(opt_kernels)

    # start = time.time()
    gradient(test_opc.m_params)
    end = time.time()
    print(end-start)
    # outer_image = write_image_file(gradient.m_image[0], MAX_DOSE)
    # inner_image = write_image_file(gradient.m_image[1], MIN_DOSE)
    # nominal_image = write_image_file(gradient.m_image[2], NOMINAL_DOSE)

    # check_equal_image("/Users/zhubinwu/research/opc-hsd/cuilt/build/pixel_statistics.txt",nominal_image[:,:,2])
    # plt.imshow(write_image_file(gradient.m_image[0], MAX_DOSE))
    # plt.savefig("outer_image_iter1.png")
    # plt.clf()
    # plt.imshow(write_image_file(gradient.m_image[1], MIN_DOSE))
    # plt.savefig("inner_image_iter1.png")
    # plt.clf()
    # plt.imshow(write_image_file(gradient.m_image[2], NOMINAL_DOSE))
    # plt.savefig("nominal_image_iter1.png")
