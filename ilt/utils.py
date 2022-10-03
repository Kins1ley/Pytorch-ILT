from constant import MASK_PRINTABLE_THRESHOLD


def is_pixel_on(mask, value=None, h_coord=None, v_coord=None):
    if value is not None:
        return value >= MASK_PRINTABLE_THRESHOLD
    elif (h_coord is not None) and (v_coord is not None):
        return mask[v_coord, h_coord] >= MASK_PRINTABLE_THRESHOLD

