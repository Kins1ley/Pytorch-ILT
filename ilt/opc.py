import sys
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapes.design import Design
from shapes.shape import Rect, Polygon

OPC_ITERATION_THRESHOLD = 20    # ILT迭代次数
OPC_TILE_SIZE = 2048 * 2048     # MASK大小
LITHOSIM_OFFSET = 512           # glp文件中坐标的原点
OPC_TILE_X = 2048   # MASK行数
OPC_TILE_Y = 2048   # MASK列数

MASK_TILE_END_X = LITHOSIM_OFFSET + 1280    # ?
MASK_TILE_END_Y = LITHOSIM_OFFSET + 1280    # ?
MASK_PRINTABLE_THRESHOLD = 0.5


class OPC(object):

    def __init__(self, design):
        self.m_design = design
        self.m_min_width = 1024
        self.m_min_height = 1024
        # self.m_minObjValue = sys.float_info.max
        # self.m_numFinalIteration = OPC_ITERATION_THRESHOLD
        self.m_targetImage = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
        self.m_mask = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)

    def rect2matrix(self, origin_x, origin_y):
        rects = self.m_design.get_rects
        for rect in rects:
            llx = rect.ll.x - origin_x
            lly = rect.ll.y - origin_y
            urx = rect.ur.x - origin_x
            ury = rect.ur.y - origin_y
            is_overlap = not (llx >= OPC_TILE_X or urx < 0 or lly >= OPC_TILE_Y or ury < 0)
            if is_overlap:
                x_bound = min(OPC_TILE_X-1, urx)
                y_bound = min(OPC_TILE_Y-1, ury)
                for x in range(max(0, llx), x_bound):
                    for y in range(max(0, lly), y_bound):
                        self.m_targetImage[y, x] = 1
            if (urx - llx) > 22:
                self.m_min_width = min(self.m_min_width, urx - llx)
            if (ury - lly) > 22:
                self.m_min_width = min(self.m_min_width, ury - lly)

    def is_pixel_value_on(self, value):
        return value >= MASK_PRINTABLE_THRESHOLD

    def matrix2rect(self, origin_x, origin_y):
        # todo: test the correctness
        rects = self.m_design.get_mask_rects
        start = -1
        for y in range(LITHOSIM_OFFSET, MASK_TILE_END_Y):
            for x in range(LITHOSIM_OFFSET, MASK_TILE_END_X):
                if not (self.is_pixel_value_on(self.m_mask[y, x])) or (x == MASK_TILE_END_X - 1):
                    if start != -1:
                        rect = Rect(origin_x+start, origin_y+y, origin_x+x-1, origin_y+y+1)
                        rects.append(rect)
                        start = -1
                else:
                    if start == -1:
                        start = x

    def init_mask(self):
        # index = 0
        self.m_mask[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = \
            self.m_targetImage[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]

        # todo: add_sraf 和 INITIAL_HAMMER_MASK

    def initialize_params(self):
        pass


if __name__ == "__main__":
    test1_design = Design("../benchmarks/M1_test1.glp")
    test_opc = OPC(test1_design)
    test_opc.rect2matrix(-LITHOSIM_OFFSET, -LITHOSIM_OFFSET)
    plt.imshow(test_opc.m_targetImage)
    plt.savefig('fig_test1.png')

