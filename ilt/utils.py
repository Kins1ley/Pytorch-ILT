from constant import MASK_PRINTABLE_THRESHOLD


def is_pixel_on(mask, value=None, h_coord=None, v_coord=None):
    if value is not None:
        return value >= MASK_PRINTABLE_THRESHOLD
    elif (h_coord is not None) and (v_coord is not None):
        return mask[v_coord, h_coord] >= MASK_PRINTABLE_THRESHOLD


def is_convex(pre_point, cur_point, next_point):
    return ((cur_point.x - pre_point.x) * (next_point.y - pre_point.y) -
            (cur_point.y - pre_point.y) * (next_point.x - pre_point.x)) > 0