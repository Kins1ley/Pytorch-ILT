import torch
from .constant import EPE_TILE_X, EPE_TILE_Y
from .constant import EPE_OFFSET_X, EPE_OFFSET_Y
from .constant import MIN_EPE_CHECK_LENGTH, EPE_CHECK_INTERVAL, EPE_CHECK_START_INTERVAL
from .epe_sample import EpeSample
from shapes import Coordinate, Polygon

class EpeChecker(object):
    orient_t = {"HORIZONTAL": 0, "VERTICAL": 1}
    direction_t = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    def __init__(self):
        self.m_valid = False
        self.m_num_epe_in = 0
        self.m_num_epe_out = 0
        self.m_polygons = None
        self.m_num_true_polygons = 0
        self.m_bimage = torch.zeros([EPE_TILE_Y, EPE_TILE_X])
        self.m_samples = []

    def set_design(self, design):
        self.m_valid = True
        polygons = design.polygons
        self.m_polygons = polygons
        self.m_num_true_polygons = len(self.m_polygons)

        rects = design.rects
        num_true_rects = design.num_true_rects
        for i in range(num_true_rects):
            ll = rects[i].ll
            ur = rects[i].ur
            polygon = Polygon()
            polygon.add_point(ll.x, ll.y)
            polygon.add_point(ur.x, ll.y)
            polygon.add_point(ur.x, ur.y)
            polygon.add_point(ll.x, ur.y)
            self.m_polygons.append(polygon)

    def set_epe_safe_region(self, epe_weight, constraint):
        if not self.m_valid:
            print("must set design before checking EPE")
        # epe_weight = torch.zeros([EPE_TILE_Y, EPE_TILE_X], dtype=torch.float64)
        for polygon in self.m_polygons:
            points = polygon.points
            pt1 = points[-1]
            for pt2 in points:
                if pt1.y == pt2.y:
                    start = pt1.x
                    end = pt2.x
                    if start > end:
                        start, end = end, start
                    for y in range(pt1.y-constraint, pt1.y+constraint+1):
                        for x in range(start-constraint, end+constraint+1):
                            epe_weight[EPE_OFFSET_Y+y, EPE_OFFSET_X+x] = 1
                else:
                    start = pt1.y
                    end = pt2.y
                    if start > end:
                        start, end = end, start
                    for y in range(start-constraint, end+constraint+1):
                        for x in range(pt1.x-constraint, pt1.x+constraint+1):
                            epe_weight[EPE_OFFSET_Y+y, EPE_OFFSET_X+x] = 1
                pt1 = pt2
        return epe_weight

    def find_sample_point(self):
        HORIZONTAL = EpeChecker.orient_t["HORIZONTAL"]
        VERTICAL = EpeChecker.orient_t["VERTICAL"]
        for polygon in self.m_polygons:
            points = polygon.points
            pt1 = points[-1]
            for pt2 in points:
                if pt1.y == pt2.y:
                    start = pt1.x
                    end = pt2.x
                    if start > end:
                        start, end = end, start
                    center = (pt1.x + pt2.x) // 2
                    if (end - start) <= MIN_EPE_CHECK_LENGTH:
                        self.m_samples.append(EpeSample(center, pt1.y, HORIZONTAL))
                    else:
                        for x in range(start+EPE_CHECK_START_INTERVAL, center+1, EPE_CHECK_INTERVAL):
                            self.m_samples.append(EpeSample(x, pt1.y, HORIZONTAL))
                        for x in range(end-EPE_CHECK_START_INTERVAL, center, -EPE_CHECK_INTERVAL):
                            self.m_samples.append(EpeSample(x, pt1.y, HORIZONTAL))
                else:
                    start = pt1.y
                    end = pt2.y
                    if start > end:
                        start, end = end, start
                    center = (pt1.y + pt2.y) // 2
                    if (end - start) <= MIN_EPE_CHECK_LENGTH:
                        self.m_samples.append(EpeSample(pt1.x, center, VERTICAL))
                    else:
                        for y in range(start+EPE_CHECK_START_INTERVAL, center+1, EPE_CHECK_INTERVAL):
                            self.m_samples.append(EpeSample(pt1.x, y, VERTICAL))
                        for y in range(end-EPE_CHECK_START_INTERVAL, center, -EPE_CHECK_INTERVAL):
                            self.m_samples.append(EpeSample(pt1.x, y, VERTICAL))

                pt1 = pt2

        print("Total number of EPE samples {}".format(str(len(self.m_samples))))
        return self.m_samples