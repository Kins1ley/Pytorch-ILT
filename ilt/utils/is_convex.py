def is_convex(pre_point, cur_point, next_point):
    return ((cur_point.x - pre_point.x) * (next_point.y - pre_point.y) -
            (cur_point.y - pre_point.y) * (next_point.x - pre_point.x)) > 0
