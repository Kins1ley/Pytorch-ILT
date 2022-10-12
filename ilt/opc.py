import sys
sys.path.append("..")
import torch
import matplotlib
import matplotlib.pyplot as plt
import time

from shapes import Design
from shapes import Rect, Polygon
from constant import OPC_TILE_X, OPC_TILE_Y
from constant import MASK_TILE_END_X, MASK_TILE_END_Y, MASKRELAX_SIGMOID_STEEPNESS, MASK_PRINTABLE_THRESHOLD
from constant import LITHOSIM_OFFSET
from constant import OPC_INITIAL_STEP_SIZE, OPC_JUMP_STEP_SIZE, GRADIENT_DESCENT_BETA
from constant import ZERO_ERROR, WEIGHT_EPE_REGION
from constant import device
from hammer import Hammer
from sraf import Sraf
from utils import is_pixel_on
from eval import EpeChecker


class OPC(object):

    def __init__(self, design, hammer=None, sraf=None):
        self.m_design = design
        self.m_min_width = 1024
        self.m_min_height = 1024
        self.m_hammer = hammer
        self.m_sraf = sraf
        # self.m_minObjValue = sys.float_info.max
        # self.m_numFinalIteration = OPC_ITERATION_THRESHOLD
        self.m_targetImage = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64, device=device)
        # M matrix
        self.m_mask = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64, device=device)
        # P matrix
        self.m_params = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64, device=device)

        # EPE checker setting
        self.m_epe_cheker = EpeChecker(device)
        self.m_epe_cheker.set_design(self.m_design)
        self.m_epe_weight = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64, device=device)
        self.m_epe_samples = None
        # step size
        self.m_step_size = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64, device=device)

    def rect2matrix(self, origin_x, origin_y):
        rects = self.m_design.rects
        for rect in rects:
            llx = rect.ll.x - origin_x
            lly = rect.ll.y - origin_y
            urx = rect.ur.x - origin_x
            ury = rect.ur.y - origin_y
            is_overlap = not (llx >= OPC_TILE_X or urx < 0 or lly >= OPC_TILE_Y or ury < 0)
            if is_overlap:
                x_bound = min(OPC_TILE_X - 1, urx)
                y_bound = min(OPC_TILE_Y - 1, ury)
                self.m_targetImage[max(0, lly):y_bound, max(0, llx):x_bound] = 1
            if (urx - llx) > 22:
                self.m_min_width = min(self.m_min_width, urx - llx)
            if (ury - lly) > 22:
                self.m_min_width = min(self.m_min_width, ury - lly)

    def matrix2rect(self, origin_x, origin_y):
        # todo: test the correctness
        rects = self.m_design.mask_rects
        start = -1
        for y in range(LITHOSIM_OFFSET, MASK_TILE_END_Y):
            for x in range(LITHOSIM_OFFSET, MASK_TILE_END_X):
                if not (is_pixel_on(self.m_mask, value=self.m_mask[y, x])) or (x == MASK_TILE_END_X - 1):
                    if start != -1:
                        rect = Rect(origin_x + start, origin_y + y, origin_x + x - 1, origin_y + y + 1)
                        rects.append(rect)
                        start = -1
                else:
                    if start == -1:
                        start = x

    def run(self):
        self.rect2matrix(-LITHOSIM_OFFSET, -LITHOSIM_OFFSET)
        self.initialize_mask()
        self.initialize_params()
        self.update_mask()

    def initialize_mask(self):
        self.m_mask[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = \
            self.m_targetImage[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]

        if self.m_hammer:
            hammer_mask = Hammer(self.m_design, self.m_mask)
            hammer_mask.initialize_mask()
            self.m_mask = hammer_mask.m_mask
        elif self.m_sraf:
            sraf_mask = Sraf(self.m_design, self.m_mask)
            sraf_mask.add_sraf()
            self.m_mask = sraf_mask.m_mask

    def update_mask(self):
        self.m_mask = torch.sigmoid(MASKRELAX_SIGMOID_STEEPNESS * self.m_params)

    def initialize_params(self):

        temp_params = torch.ones([MASK_TILE_END_Y-LITHOSIM_OFFSET,
                                  MASK_TILE_END_X-LITHOSIM_OFFSET], dtype=torch.float64)
        temp_params[self.m_mask[LITHOSIM_OFFSET:MASK_TILE_END_Y,
                                LITHOSIM_OFFSET:MASK_TILE_END_X] < MASK_PRINTABLE_THRESHOLD] = -1
        self.m_params[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = temp_params

        # the above implementation is the same as the below one
        # temp = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
        # for y in range(LITHOSIM_OFFSET, MASK_TILE_END_Y):
        #     for x in range(LITHOSIM_OFFSET, MASK_TILE_END_X):
        #         if is_pixel_on(self.m_mask, h_coord=x, v_coord=y):
        #             temp[y, x] = 1
        #         else:
        #             temp[y, x] = -1
        # print(self.m_params.equal(temp))

    def determine_epe_weight(self, num_iteration=1):
        if num_iteration == 1:
            self.m_epe_cheker.set_epe_safe_region(self.m_epe_weight, constraint=10)
            self.m_epe_weight[self.m_epe_weight < ZERO_ERROR] = WEIGHT_EPE_REGION

            # the above implementation is the same as the below one

            # temp = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
            # self.m_epe_cheker.set_epe_safe_region(temp, constraint=10)
            # for y in range(OPC_TILE_Y):
            #     for x in range(OPC_TILE_X):
            #         if temp[y, x] < ZERO_ERROR:
            #             temp[y, x] = 0.5
            # print(self.m_epe_weight.equal(temp))

        elif num_iteration == 5:
            self.m_epe_weight[:, :] = 1
            # for y in range(OPC_TILE_Y):
            #     for x in range(OPC_TILE_X):
            #         self.m_epe_weight[y, x] = 1

    def determine_const_step_size(self, num_iteration):
        self.m_step_size[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = OPC_INITIAL_STEP_SIZE


if __name__ == "__main__":
    # matplotlib.use('TkAgg')
    test_design = Design("../benchmarks/M1_test1" + ".glp")
    test_opc = OPC(test_design, hammer=1, sraf=0)
    # start = time.time()
    test_opc.run()
    # end = time.time()
    # print(end-start)
    plt.imshow(test_opc.m_mask)
    plt.savefig('init_mask.png')

