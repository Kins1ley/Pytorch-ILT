from constant import LITHOSIM_OFFSET
from constant import OPC_TILE_X, OPC_TILE_Y
from constant import OPC_SPACE_SRAF, OPC_WIDTH_SRAF, OPC_SPACE_FORBID_SRAF
from utils import is_pixel_on


class Sraf(object):

    def __init__(self, design, mask):
        self.m_design = design
        self.m_mask = mask

        self.orient = {"HORIZONTAL": 0, "VERTICAL": 1}

    def draw_sraf(self, x, y, length, orient):
        if orient == self.orient["HORIZONTAL"]:
            llx = x
            if is_pixel_on(self.m_mask, h_coord=x, v_coord=y-OPC_SPACE_SRAF):
                lly = y + OPC_SPACE_SRAF
                forbid_pos = lly + OPC_WIDTH_SRAF + OPC_SPACE_FORBID_SRAF
            else:
                lly = max(LITHOSIM_OFFSET, y-OPC_SPACE_SRAF-OPC_WIDTH_SRAF)
                forbid_pos = max(LITHOSIM_OFFSET, lly-OPC_SPACE_FORBID_SRAF)

            is_valid = True
            from_pos = min(lly, forbid_pos)
            to_pos = max(lly, forbid_pos)
            for j in range(from_pos, to_pos+1):
                i = llx
                while (i < (llx + length)) and is_valid:
                    if is_pixel_on(self.m_mask, h_coord=i, v_coord=j):
                        is_valid = False
                    i += 1

            if is_valid:
                self.m_mask[lly:lly+OPC_WIDTH_SRAF, llx:llx+length] = 1
                # for j in range(lly, lly+OPC_WIDTH_SRAF):
                #     for i in range(llx, llx+length):
                #         self.m_mask[j, i] = 1

        else:
            lly = y
            if is_pixel_on(self.m_mask, h_coord=x-OPC_SPACE_SRAF, v_coord=y):
                llx = x + OPC_SPACE_SRAF
                forbid_pos = llx + OPC_WIDTH_SRAF + OPC_SPACE_FORBID_SRAF
            else:
                llx = max(LITHOSIM_OFFSET, x-OPC_SPACE_SRAF-OPC_WIDTH_SRAF)
                forbid_pos = max(LITHOSIM_OFFSET, llx-OPC_SPACE_FORBID_SRAF)

            is_valid = True
            from_pos = min(llx, forbid_pos)
            to_pos = max(llx, forbid_pos)
            j = lly
            while (j < (lly + length)) and is_valid:
                for i in range(from_pos, to_pos+1):
                    if is_pixel_on(self.m_mask, h_coord=i, v_coord=j):
                        is_valid = False
                j += 1

            if is_valid:
                self.m_mask[lly:lly+length, llx:llx+OPC_WIDTH_SRAF] = 1
                # for j in range(lly, lly+length):
                #     for i in range(llx, llx+OPC_WIDTH_SRAF):
                #         self.m_mask[j, i] = 1

    def add_sraf(self):
        # todo: test the correctness
        rects = self.m_design.rects
        # orient = {"HORIZONTAL": 0, "VERTICAL":1}
        for rect in rects:
            llx = rect.ll.x + LITHOSIM_OFFSET
            lly = rect.ll.y + LITHOSIM_OFFSET
            urx = rect.ur.x + LITHOSIM_OFFSET
            ury = rect.ur.y + LITHOSIM_OFFSET
            is_overlap = not (llx >= OPC_TILE_X or urx < 0 or lly >= OPC_TILE_Y or ury < 0)
            if is_overlap:
                self.draw_sraf(llx, lly, urx - llx, self.orient["HORIZONTAL"])
                self.draw_sraf(llx, ury, urx - llx, self.orient["HORIZONTAL"])
                self.draw_sraf(llx, lly, ury - lly, self.orient["VERTICAL"])
                self.draw_sraf(urx, lly, ury - lly, self.orient["VERTICAL"])

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
                    if (i + 1) < length:
                        next_point = points[i+1]
                    else:
                        next_point = points[0]

                    if cur_point.y == next_point.y:
                        self.draw_sraf(LITHOSIM_OFFSET + min(cur_point.x, next_point.x), LITHOSIM_OFFSET + cur_point.y,
                                       abs(cur_point.x - next_point.x), self.orient["HORIZONTAL"])

                    else:
                        self.draw_sraf(LITHOSIM_OFFSET + cur_point.x, LITHOSIM_OFFSET + min(cur_point.y, next_point.y),
                                       abs(cur_point.y - next_point.y), self.orient["VERTICAL"])


