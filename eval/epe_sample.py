from eval import EPE_OFFSET_X, EPE_OFFSET_Y

class EpeSample(object):
    def __init__(self, x, y, orient):
        self.m_orient = orient
        self.m_x = EPE_OFFSET_X + x
        self.m_y = EPE_OFFSET_Y + y
        self.m_num_violation = 0
        self.m_sig_product = None

    @property
    def x(self):
        return self.m_x

    @property
    def y(self):
        return self.m_y

    @property
    def orient(self):
        return self.m_orient

    @property
    def sig_product(self):
        return self.m_sig_product

    @sig_product.setter
    def sig_product(self, value):
        self.m_sig_product = value

    # def set_position(self, from_pos):
    #     if self.m_orient == EpeChecker.orient_t["HORIZONTAL"]:
    #         to_pos[] = from_pos



