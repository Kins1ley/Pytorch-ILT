# import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
from shapes.design import Design
from shapes.shape import Rect, Polygon
from constant import OPC_TILE_X, OPC_TILE_Y, OPC_LENGTH_CORNER_RESHAPE
from constant import MASK_PRINTABLE_THRESHOLD, MASK_TILE_END_X, MASK_TILE_END_Y
from constant import LITHOSIM_OFFSET


class OPC(object):

    def __init__(self, design, init_hammer):
        self.m_design = design
        self.m_min_width = 1024
        self.m_min_height = 1024
        self.init_hammer = init_hammer
        # self.m_minObjValue = sys.float_info.max
        # self.m_numFinalIteration = OPC_ITERATION_THRESHOLD
        self.m_targetImage = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
        self.m_mask = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)
        self.m_params = torch.zeros([OPC_TILE_Y, OPC_TILE_X], dtype=torch.float64)

    def rect2matrix(self, origin_x, origin_y):
        rects = self.m_design.get_rects
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

    def is_pixel_on(self, value=None, h_coord=None, v_coord=None):
        if value is not None:
            return value >= MASK_PRINTABLE_THRESHOLD
        elif (h_coord is not None) and (v_coord is not None):
            return self.m_mask[v_coord, h_coord] >= MASK_PRINTABLE_THRESHOLD

    def matrix2rect(self, origin_x, origin_y):
        # todo: test the correctness
        rects = self.m_design.get_mask_rects
        start = -1
        for y in range(LITHOSIM_OFFSET, MASK_TILE_END_Y):
            for x in range(LITHOSIM_OFFSET, MASK_TILE_END_X):
                if not (self.is_pixel_on(value=self.m_mask[y, x])) or (x == MASK_TILE_END_X - 1):
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

        if self.init_hammer:
            self.initialize_hammer_mask()
        # todo: add_sraf 和 INITIAL_HAMMER_MASK

    def draw_hammer(self, x, y, value):
        # todo: test the correctness
        # for j in range(y - OPC_LENGTH_CORNER_RESHAPE, y + OPC_LENGTH_CORNER_RESHAPE):
        #     for i in range(x - OPC_LENGTH_CORNER_RESHAPE, x + OPC_LENGTH_CORNER_RESHAPE):
        self.m_mask[y - OPC_LENGTH_CORNER_RESHAPE:y + OPC_LENGTH_CORNER_RESHAPE,
                    x - OPC_LENGTH_CORNER_RESHAPE:x + OPC_LENGTH_CORNER_RESHAPE] = value

    def is_convex(self, pre_point, cur_point, next_point):
        # todo: test the correctness
        return ((cur_point.x - pre_point.x) * (next_point.y - pre_point.y) -
                (cur_point.y - pre_point.y) * (next_point.x - pre_point.x)) > 0

    def initialize_hammer_mask(self):
        # todo: test the correctness
        rects = self.m_design.get_rects
        num_true_rects = self.m_design.get_num_true_rects
        for i in range(num_true_rects):
            rect = rects[i]
            llx = rect.ll.x + LITHOSIM_OFFSET
            lly = rect.ll.y + LITHOSIM_OFFSET
            urx = rect.ur.x + LITHOSIM_OFFSET
            ury = rect.ur.y + LITHOSIM_OFFSET
            is_overlap = not (llx >= OPC_TILE_X or urx < 0 or lly >= OPC_TILE_Y or ury < 0)
            if is_overlap:
                self.draw_hammer(llx, lly, 1)
                self.draw_hammer(urx, lly, 1)
                self.draw_hammer(urx, ury, 1)
                self.draw_hammer(llx, ury, 1)

        polygons = self.m_design.get_polygons
        for polygon in polygons:
            points = polygon.get_points
            length = len(points)
            for i in range(length):
                cur_point = points[i]
                x = cur_point.x + LITHOSIM_OFFSET
                y = cur_point.y + LITHOSIM_OFFSET
                is_overlap = not (x >= OPC_TILE_X or x < 0 or y >= OPC_TILE_Y or y < 0)
                if is_overlap:
                    if (i - 1) >= 0:
                        pre_point = points[i - 1]
                    else:
                        pre_point = points[length - 1]

                    if (i + 1) < length:
                        next_point = points[i + 1]
                    else:
                        next_point = points[0]

                    if self.is_convex(pre_point, cur_point, next_point):
                        self.draw_hammer(x, y, 1)

                    else:
                        self.draw_hammer(x, y, 0)

    def initialize_params(self):
        # todo: test the correctness
        for y in range(LITHOSIM_OFFSET, MASK_TILE_END_Y):
            for x in range(LITHOSIM_OFFSET, MASK_TILE_END_X):
                if self.is_pixel_on(h_coord=x, v_coord=y):
                    self.m_params[y, x] = 1
                else:
                    self.m_params[y, x] = -1


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    test1_design = Design("../benchmarks/M1_test1.glp")
    test_opc = OPC(test1_design, 1)
    test_opc.run()
    # plt.imshow(test_opc.m_targetImage)
    # plt.savefig('fig_test1.png')

