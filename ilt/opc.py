# import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
from shapes import Design
from shapes import Rect, Polygon
from constant import OPC_TILE_X, OPC_TILE_Y
from constant import MASK_TILE_END_X, MASK_TILE_END_Y
from constant import LITHOSIM_OFFSET
from hammer import Hammer
from sraf import Sraf
from utils import is_pixel_on


class OPC(object):

    def __init__(self, design, hammer=None, sraf=None):
        self.m_design = design
        self.m_min_width = 1024
        self.m_min_height = 1024
        self.m_hammer = hammer
        self.m_sraf = sraf
        # self.m_minObjValue = sys.float_info.max
        # self.m_numFinalIteration = OPC_ITERATION_THRESHOLD
        self.m_targetImage = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
        self.m_mask = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
        self.m_params = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)

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
                for x in range(max(0, llx), x_bound):
                    for y in range(max(0, lly), y_bound):
                        self.m_targetImage[y, x] = 1
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

    def initialize_params(self):
        for y in range(LITHOSIM_OFFSET, MASK_TILE_END_Y):
            for x in range(LITHOSIM_OFFSET, MASK_TILE_END_X):
                if is_pixel_on(self.m_mask, h_coord=x, v_coord=y):
                    self.m_params[y, x] = 1
                else:
                    self.m_params[y, x] = -1


if __name__ == "__main__":
    import time
    matplotlib.use('TkAgg')
    for i in range(1, 11):
        test_design = Design("../benchmarks/M1_test" + str(i) + ".glp")
        test_opc = OPC(test_design, hammer=0, sraf=1)
        # start = time.time()
        test_opc.run()
        # end = time.time()
        # print(end-start)
        plt.imshow(test_opc.m_mask)
        # plt.show()
        plt.savefig('add_sraf_mask' + str(i) + '.png')

