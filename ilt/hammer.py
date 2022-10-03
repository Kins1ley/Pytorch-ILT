import torch
from constant import OPC_TILE_X, OPC_TILE_Y, OPC_LENGTH_CORNER_RESHAPE
from constant import LITHOSIM_OFFSET


class Hammer(object):
    def __init__(self, design, mask):
        self.m_design = design
        self.m_mask = mask

    def draw_hammer(self, x, y, value):
        # todo: test the correctness
        # for j in range(y - OPC_LENGTH_CORNER_RESHAPE, y + OPC_LENGTH_CORNER_RESHAPE):
        #     for i in range(x - OPC_LENGTH_CORNER_RESHAPE, x + OPC_LENGTH_CORNER_RESHAPE):
        self.m_mask[y-OPC_LENGTH_CORNER_RESHAPE : y+OPC_LENGTH_CORNER_RESHAPE,
                    x-OPC_LENGTH_CORNER_RESHAPE : x+OPC_LENGTH_CORNER_RESHAPE] = value

    def initialize_mask(self):
        # todo: test the correctness
        rects = self.m_design.rects
        num_true_rects = self.m_design.num_true_rects
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

        polygons = self.m_design.polygons
        for polygon in polygons:
            points = polygon.points
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

    def is_convex(self, pre_point, cur_point, next_point):
        # todo: test the correctness
        return ((cur_point.x - pre_point.x) * (next_point.y - pre_point.y) -
                (cur_point.y - pre_point.y) * (next_point.x - pre_point.x)) > 0