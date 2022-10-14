import torch
import torch.nn as nn
import time
import matplotlib
import matplotlib.pyplot as plt
import sys
import platform
sys.path.append("..")

from kernels import Kernel
from shapes import Design

import simulator
from opc import OPC
from constant import OPC_TILE_X, OPC_TILE_Y
from constant import MAX_DOSE, MIN_DOSE, NOMINAL_DOSE
from constant import MASKRELAX_SIGMOID_STEEPNESS, PHOTORISIST_SIGMOID_STEEPNESS, TARGET_INTENSITY
from constant import device
from constant import LITHOSIM_OFFSET, MASK_TILE_END_X, MASK_TILE_END_Y
from constant import WEIGHT_PVBAND, WEIGHT_REGULARIZATION
from logger import logger
from utils import write_image_file
from utils import check_equal_image


class GradientBlock(nn.Module):
    def __init__(self, kernels, design):
        super(GradientBlock, self).__init__()
        self.design = design
        # init mask
        self.update_mask = nn.Sigmoid()
        self.litho_sigmoid = nn.Sigmoid()
        self.kernels = kernels
        self.filter = torch.zeros([OPC_TILE_Y, OPC_TILE_X]).to(device)
        self.filter[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1
        self.target_image = self.design.m_target_image
        self.epe_weight = self.design.m_epe_weight
        self.simulator = simulator.Simulator()

    def forward(self, params, num_iteration):
        gradient = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64).to(device)
        image = [None, None, None]
        term = [None, None, None]
        cpx_term = [None, None, None]
        mask = self.update_mask(MASKRELAX_SIGMOID_STEEPNESS * params) * self.filter
        image[0] = self.simulator.simulate_image(mask, self.kernels["focus"].kernels,
                                                 self.kernels["focus"].scales, MAX_DOSE, 15)
        image[1] = self.simulator.simulate_image(mask, self.kernels["defocus"].kernels,
                                                 self.kernels["defocus"].scales, MIN_DOSE, 15)
        image[2] = self.simulator.simulate_image(mask, self.kernels["focus"].kernels,
                                                 self.kernels["focus"].scales, NOMINAL_DOSE, 15)
        pvband = self.simulator.calculate_pvband(image[1], image[0])
        print("pvband", pvband)
        # start = time.time()
        epe_convergence = self.design.m_epe_checker.run(image[2])
        # end = time.time()
        # print("epe check time", end-start)

        image[0] = self.litho_sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[0] - TARGET_INTENSITY))
        image[1] = self.litho_sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[1] - TARGET_INTENSITY))
        image[2] = self.litho_sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[2] - TARGET_INTENSITY))

        # m_term[0]: (Znom - Zt)^3 * Znom * (1-Znom)
        term[0] = torch.pow(image[2] - self.target_image, 3) * image[2] * (1 - image[2])

        # conv(M, conj(Hnom))
        cpx_term[0] = self.simulator.convolve_image(self.kernels["combo CT focus"].kernels,
                                                    self.kernels["combo CT focus"].scales,
                                                    MAX_DOSE, 1, mask=mask)
        # (Znom - Zt)^3 * Znom * (1-Znom) * conv(M, conj(Hnom))
        cpx_term[0] = torch.mul(cpx_term[0], term[0])

        # conv(Hnom, (Znom - Zt)^3 * Znom * (1-Znom) * conv(M, conj(Hnom)))
        cpx_term[1] = self.simulator.convolve_image(self.kernels["combo focus"].kernels,
                                                    self.kernels["combo focus"].scales,
                                                    MAX_DOSE, 1, matrix=cpx_term[0])

        # conv(M, Hnom)
        cpx_term[0] = self.simulator.convolve_image(self.kernels["combo focus"].kernels,
                                                    self.kernels["combo focus"].scales,
                                                    MAX_DOSE, 1, mask=mask)

        # (Znom - Zt)^3 * Znom * (1-Znom) * conv(M, Hnom)
        cpx_term[0] = torch.mul(cpx_term[0], term[0])

        # conv(conj(Hnom), (Znom - Zt)^3 * Znom * (1-Znom) * conv(M, Hnom))
        cpx_term[2] = self.simulator.convolve_image(self.kernels["combo CT focus"].kernels,
                                                    self.kernels["combo CT focus"].scales,
                                                    MAX_DOSE, 1, matrix=cpx_term[0])

        cpx_term[0] = cpx_term[1] + cpx_term[2]

        # 4 * theta_z * theta_m * cpx_term[0]
        gradient = 4 * PHOTORISIST_SIGMOID_STEEPNESS * MASKRELAX_SIGMOID_STEEPNESS * self.epe_weight * cpx_term[0].real

        # m_term[0]: (Z_defocus - Zt) * Z_defocus * (1-Z_defocus)
        term[0] = (image[1] - self.target_image) * image[1] * (1 - image[1])

        # conv(M, conj(H_defocus))
        cpx_term[0] = self.simulator.convolve_image(self.kernels["combo CT defocus"].kernels,
                                                    self.kernels["combo CT defocus"].scales,
                                                    MIN_DOSE, 1, mask=mask)

        # (Z_defocus - Zt) * Z_defocus * (1-Z_defocus) * conv(M, conj(H_defocus))
        cpx_term[0] = torch.mul(term[0], cpx_term[0])

        # conv(H_defocus, (Z_defocus - Zt) * Z_defocus * (1-Z_defocus) * conv(M, conj(H_defocus)))
        cpx_term[1] = self.simulator.convolve_image(self.kernels["combo defocus"].kernels,
                                                    self.kernels["combo defocus"].scales,
                                                    MIN_DOSE, 1, matrix=cpx_term[0])

        # conv(M, H_defocus)
        cpx_term[0] = self.simulator.convolve_image(self.kernels["combo defocus"].kernels,
                                                    self.kernels["combo defocus"].scales,
                                                    MIN_DOSE, 1, mask=mask)

        # (Z_defocus - Zt) * Z_defocus * (1-Z_defocus) * conv(M, H_defocus)
        cpx_term[0] = torch.mul(term[0], cpx_term[0])

        # conv(conj(H_defocus), (Z_defocus - Zt) * Z_defocus * (1-Z_defocus) * conv(M, H_defocus))
        cpx_term[2] = self.simulator.convolve_image(self.kernels["combo CT defocus"].kernels,
                                                    self.kernels["combo CT defocus"].scales,
                                                    MIN_DOSE, 1, matrix=cpx_term[0])
        cpx_term[0] = cpx_term[1] + cpx_term[2]

        pvb_gradient_constant = WEIGHT_PVBAND * 2 * PHOTORISIST_SIGMOID_STEEPNESS * MASKRELAX_SIGMOID_STEEPNESS
        discrete_penalty = WEIGHT_REGULARIZATION * (-8 * mask + 4)
        gradient = mask * (1 - mask) * (gradient + pvb_gradient_constant * cpx_term[0].real +
                                        MASKRELAX_SIGMOID_STEEPNESS * discrete_penalty) * self.filter

        diff_target = image[2] - self.target_image
        diff_image = image[0] - image[1]
        step_size = self.design.determine_step_size_backtrack(num_iteration, self.filter, diff_target,
                                                              diff_image, discrete_penalty)
        self.design.update_convergence(diff_target, diff_image, discrete_penalty, epe_convergence, pvband)
        self.design.keep_best_result(mask)
        params[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] -= \
            step_size[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] * \
            gradient[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
        # return params
        # maxg = torch.max(gradient)
        # ming = torch.min(gradient)
        # print("maxg: {}, ming: {}".format(maxg, ming))
        return params
        # print("obj_convergence:", self.design.m_obj_convergence[0])
        # print(step_size[1024, 1020:1030])


if __name__ == "__main__":
    torch.set_printoptions(precision=7)
    # kernel setting
    if platform.system() == "Darwin":
        matplotlib.use('TkAgg')
    # start = time.time()
    defocus_flag = 1
    conjuncture_flag = 1
    combo_flag = 1
    # 4 kinds of kernels: focus, defocus, combo focus, combo CT focus
    opt_kernels = {"focus": Kernel(35, 35), "defocus": Kernel(35, 35, defocus=defocus_flag),
                   # "CT focus": Kernel(35, 35, conjuncture=conjuncture_flag),
                   # "CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag),
                   "combo focus": Kernel(35, 35, combo=combo_flag),
                   "combo defocus": Kernel(35, 35, defocus=defocus_flag, combo=combo_flag),
                   "combo CT focus": Kernel(35, 35, conjuncture=conjuncture_flag, combo=combo_flag),
                   "combo CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag,
                                              combo=combo_flag)}
    # init design file and init params
    m1_test = Design("../benchmarks/M1_test1" + ".glp")
    test_design = OPC(m1_test, hammer=1, sraf=0)
    test_design.run()
    # init the gradient block
    update_block = GradientBlock(opt_kernels, test_design)
    # calculate the simulated image of the init_params
    start = time.time()
    params = test_design.m_params
    for i in range(20):
        params = update_block(params, i+1)
    end = time.time()
    print("20 iteration time:", end-start)
    print(test_design.m_min_score)

    # check_equal_image("/Users/zhubinwu/research/opc-hsd/cuilt/build/pixel_statistics.txt",nominal_image[:,:,2])
    # plt.imshow(write_image_file(image[0], MAX_DOSE))
    # plt.savefig("outer_image_iter1.png")
    # plt.clf()
    # plt.imshow(write_image_file(image[1], MIN_DOSE))
    # plt.savefig("inner_image_iter1.png")
    # plt.clf()
    # plt.imshow(write_image_file(image[2], NOMINAL_DOSE))
    # plt.savefig("nominal_image_iter1.png")
