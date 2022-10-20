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
from constant import OPC_ITERATION_THRESHOLD
from constant import OPC_TILE_X, OPC_TILE_Y
from constant import MAX_DOSE, MIN_DOSE, NOMINAL_DOSE
from constant import MASKRELAX_SIGMOID_STEEPNESS, PHOTORISIST_SIGMOID_STEEPNESS, TARGET_INTENSITY
from constant import device
from constant import LITHOSIM_OFFSET, MASK_TILE_END_X, MASK_TILE_END_Y
from constant import WEIGHT_PVBAND, WEIGHT_REGULARIZATION
from logger import get_logger
import logging
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
        self.filter = self.design.filter
        self.target_image = self.design.m_target_image
        self.epe_weight = self.design.m_epe_weight
        self.simulator = simulator.Simulator()

    def forward(self, params, num_iteration):
        logger.debug("Iteration {}".format(num_iteration))
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
        logger.debug("pvband before gradient descent in iteration {} : {}".format(num_iteration, pvband))
        # start = time.time()
        epe_convergence = self.design.m_epe_checker.run(image[2])
        logger.debug("epe before gradient descent in iteration {} : {}".format(num_iteration, epe_convergence))
        # end = time.time()
        # print("epe check time", end-start)

        image[0] = self.litho_sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[0] - TARGET_INTENSITY))
        image[1] = self.litho_sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[1] - TARGET_INTENSITY))
        image[2] = self.litho_sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[2] - TARGET_INTENSITY))
        criterion = nn.MSELoss(reduction="sum")
        loss = criterion(image[2], self.target_image)
        logger.debug("loss: {}".format(loss.item()))
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
        # gradient = mask * (1 - mask) * (gradient + pvb_gradient_constant * cpx_term[0].real +
        #                                 MASKRELAX_SIGMOID_STEEPNESS * discrete_penalty) * self.filter
        gradient = mask * (1 - mask) * (gradient + pvb_gradient_constant * cpx_term[0].real +
                                        MASKRELAX_SIGMOID_STEEPNESS * discrete_penalty) * self.filter
        # gradient = mask * (1-mask) * gradient
        diff_target = image[2] - self.target_image
        diff_image = image[0] - image[1]
        step_size = self.design.determine_step_size_backtrack(num_iteration, diff_target,
                                                              diff_image, discrete_penalty)
        # step_size = self.design.determine_const_step_size(num_iteration)
        self.design.update_convergence(diff_target, diff_image, discrete_penalty, epe_convergence, pvband)
        self.design.keep_best_result(mask)
        params[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] -= \
            step_size[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] * \
            gradient[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
        # params -= step_size * gradient
        self.design.exit_iteration(gradient, num_iteration)
        # return params
        # maxg = torch.max(gradient)
        # ming = torch.min(gradient)
        # print("maxg: {}, ming: {}".format(maxg, ming))
        return params
        # print("obj_convergence:", self.design.m_obj_convergence[0])
        # print(step_size[1024, 1020:1030])


def ilt(opc_kernels, design, update_block):
    # init the gradient block
    # update_block = GradientBlock(opc_kernels, design)
    # OPC_ITERATION_THRESHOLD = 2
    for num_iteration in range(1, OPC_ITERATION_THRESHOLD + 1):
        update_block(design.m_params, num_iteration)
        if design.exit_flag:
            break
    design.update_mask()
    # test_design.binary_mask()
    image = [None, None, None]
    image[0] = update_block.simulator.simulate_image(design.m_mask, opc_kernels["focus"].kernels,
                                                         opc_kernels["focus"].scales, MAX_DOSE, 15)
    image[1] = update_block.simulator.simulate_image(design.m_mask, opc_kernels["defocus"].kernels,
                                                     opc_kernels["defocus"].scales, MIN_DOSE, 15)
    image[2] = update_block.simulator.simulate_image(design.m_mask, opc_kernels["focus"].kernels,
                                                     opc_kernels["focus"].scales, NOMINAL_DOSE, 15)
    pvband = update_block.simulator.calculate_pvband(image[1], image[0])
    logger.debug("Check the final pvband and the EPE value")
    logger.debug("pvband after the whole iterations {}".format(pvband))
    # start = time.time()
    epe_convergence = test_design.m_epe_checker.run(image[2])
    image[0] = torch.sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[0] - TARGET_INTENSITY))
    image[1] = torch.sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[1] - TARGET_INTENSITY))
    image[2] = torch.sigmoid(PHOTORISIST_SIGMOID_STEEPNESS * (image[2] - TARGET_INTENSITY))
    discrete_penalty = WEIGHT_REGULARIZATION * (-8 * test_design.m_mask + 4)
    diff_target = image[2] - test_design.m_target_image
    diff_image = image[0] - image[1]
    test_design.update_convergence(diff_target, diff_image, discrete_penalty, epe_convergence, pvband)
    logger.debug("Final Object Value: {}".format(test_design.m_obj_convergence[-1]))
    test_design.restore_best_result()


def draw_final(design, update_block):
    design.binary_mask()
    image = [None, None, None]
    image[0] = update_block.simulator.simulate_image(design.bmask, opc_kernels["focus"].kernels,
                                                     opc_kernels["focus"].scales, MAX_DOSE, 24)
    image[1] = update_block.simulator.simulate_image(design.bmask, opc_kernels["defocus"].kernels,
                                                     opc_kernels["defocus"].scales, MIN_DOSE, 24)
    image[2] = update_block.simulator.simulate_image(design.bmask, opc_kernels["focus"].kernels,
                                                     opc_kernels["focus"].scales, NOMINAL_DOSE, 24)
    plt.imshow(write_image_file(image[0], MAX_DOSE))
    plt.savefig("outer_image_iter_final.png")
    plt.clf()
    plt.imshow(write_image_file(image[1], MIN_DOSE))
    plt.savefig("inner_image_iter_final.png")
    plt.clf()
    plt.imshow(write_image_file(image[2], NOMINAL_DOSE))
    plt.savefig("nominal_image_iter_final.png")

if __name__ == "__main__":
    # logging setting
    # logger = get_logger(__name__)
    torch.set_printoptions(precision=7)
    # kernel setting
    if platform.system() == "Darwin":
        matplotlib.use('TkAgg')
    # start = time.time()
    defocus_flag = 1
    conjuncture_flag = 1
    combo_flag = 1
    # 4 kinds of kernels: focus, defocus, combo focus, combo CT focus
    opc_kernels = {"focus": Kernel(35, 35), "defocus": Kernel(35, 35, defocus=defocus_flag),
                   # "CT focus": Kernel(35, 35, conjuncture=conjuncture_flag),
                   # "CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag),
                   "combo focus": Kernel(35, 35, combo=combo_flag),
                   "combo defocus": Kernel(35, 35, defocus=defocus_flag, combo=combo_flag),
                   "combo CT focus": Kernel(35, 35, conjuncture=conjuncture_flag, combo=combo_flag),
                   "combo CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag,
                                              combo=combo_flag)}
    # init design file and init params
    # start = time.time()
    # for i in range(1, 11):
    i = 10
    logger = get_logger(__name__, "experiment/case" + str(i) + ".txt")
    m1_test = Design("../benchmarks/M1_test" + str(i) + ".glp")
    test_design = OPC(m1_test, hammer=1, sraf=0)
    update_block = GradientBlock(opc_kernels, test_design)
    logger.info("start opc")
    test_design.run()
    # start = time.time()
    ilt(opc_kernels, test_design, update_block)
    # end = time.time()
    # logger.info("ilt time {}".format(end - start))
    # # print("ilt time {}".format(end - start))
    # draw_final(test_design, update_block)
    # print(end-start)
    # check_equal_image("/Users/zhubinwu/research/opc-hsd/cuilt/build/pixel_statistics.txt",nominal_image[:,:,2])
    # plt.imshow(write_image_file(image[0], MAX_DOSE))
    # plt.savefig("outer_image_iter_final.png")
    # plt.clf()
    # plt.imshow(write_image_file(image[1], MIN_DOSE))
    # plt.savefig("inner_image_iter_final.png")
    # plt.clf()
    # plt.imshow(write_image_file(image[2], NOMINAL_DOSE))
    # plt.savefig("nominal_image_iter_final.png")
