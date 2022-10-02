
class Coordinate(object):

    def __init__(self, coordx, coordy):
        self.x = coordx
        self.y = coordy

    def __str__(self):
        return str("{} {} ".format(str(self.x), str(self.y)))


class Rect(object):

    def __init__(self, llx, lly, urx, ury):
        self.m_ll = Coordinate(llx, lly)
        self.m_ur = Coordinate(urx, ury)

    @property
    def ll(self):
        return self.m_ll

    @property
    def ur(self):
        return self.m_ur

    def __str__(self):
        return str("{}{} {}\n".
                   format(str(self.m_ll), str(self.m_ur.x - self.m_ll.x),
                          str(self.m_ur.y - self.m_ll.y)))


class Polygon(object):

    def __init__(self):
        self.m_points = []  # vector<Coordinate *>
        self.m_convertedRects = []

    def add_point(self, x, y):
        coord = Coordinate(x, y)
        self.m_points.append(coord)

    def __str__(self):
        string = ""
        for coord in self.m_points:
            string += str(coord)
        string += "\n"
        return string

    @property
    def get_points(self):
        return self.m_points

    def convert_rect(self):
        h_level = set()
        v_level = set()
        num_pt = len(self.m_points)
        from_pt = self.m_points[-1]
        # for p in self.m_points:
        #     print(p)

        for i in range(num_pt):
            to_pt = self.m_points[i]
            if from_pt.x == to_pt.x:
                v_level.add(from_pt.x)
                # print(from_pt.x)
            elif from_pt.y == to_pt.y:
                h_level.add(from_pt.y)
                # print(from_pt.y)
            else:
                print("diagonal edge in polygon\n")
            from_pt = to_pt
        #
        h_level = sorted(h_level)
        v_level = sorted(v_level)

        for i in range(len(v_level) - 1):
            v_it = list(v_level)[i]
            v_next = list(v_level)[i + 1]
            for j in range(len(h_level) - 1):
                h_it = list(h_level)[j]
                h_next = list(h_level)[j + 1]
                corner1 = Coordinate(v_it, h_it)
                corner2 = Coordinate(v_next, h_next)
                center = Coordinate((corner1.x + corner2.x) / 2,
                                    (corner1.y + corner2.y) / 2)
                valid = self.point_in_polygon(center.x, center.y)
                if valid:
                    assert (corner1.x < corner2.x) and (corner1.y < corner2.y)
                    rect = Rect(corner1.x, corner1.y, corner2.x, corner2.y)
                    self.m_convertedRects.append(rect)

        # for t in self.m_convertedRects:
        #     print(t)

        return self.m_convertedRects

    def point_in_polygon(self, x, y):
        num_pt = len(self.m_points)
        from_pt = self.m_points[num_pt - 1]
        is_odd_nodes = False
        for i in range(num_pt):
            to_pt = self.m_points[i]
            from_x, from_y = from_pt.x, from_pt.y
            to_x, to_y = to_pt.x, to_pt.y
            if ((from_y < y <= to_y) or (to_y < y <= from_y)) and \
                    from_x + (y - from_y) / (to_y - from_y) * (to_x - from_x) < x:
                is_odd_nodes = not is_odd_nodes
            from_pt = to_pt
        return is_odd_nodes
