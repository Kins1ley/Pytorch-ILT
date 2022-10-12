import torch
import time
import sys
sys.path.append("..")
from ilt.constant import device
# from dumpkernel import DumpKernel

class Kernel(object):
    def __init__(self, knx, kny, defocus=0, conjuncture=0, combo=0):
        self.knx = knx
        self.kny = kny
        # self.device = device
        self.flag_defocus = defocus
        self.conjuncture = conjuncture
        self.combo = combo

        # print(self.kernel_file())
        # print(self.scale_file())
        self.kernels = torch.load(self.kernel_file(), map_location=device)
        self.scales = torch.load(self.scale_file(), map_location=device)

    def kernel_file(self):
        file_name = ""
        if self.flag_defocus:
            file_name = "defocus" + file_name
        else:
            file_name = "focus" + file_name

        if self.conjuncture:
            file_name = "ct_" + file_name

        if self.combo:
            file_name = "combo_" + file_name

        file_name = "../kernels/kernels/" + file_name + ".pt"
        return file_name

    def scale_file(self):
        file_name = "../kernels/scales/"
        if self.combo:
            return file_name + "combo.pt"
        else:
            if self.flag_defocus:
                return file_name + "defocus.pt"
            else:
                return file_name + "focus.pt"


if __name__ == "__main__":
    defocus_flag = 1
    conjuncture_flag = 1
    combo_flag = 1
    start = time.time()
    opt_kernels = {"focus": Kernel(35, 35), "defocus": Kernel(35, 35, defocus=defocus_flag),
                   "CT focus": Kernel(35, 35, conjuncture=conjuncture_flag),
                   "CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag)}

    combo_kernels = {"combo focus": Kernel(35, 35, combo=combo_flag),
                     "combo defocus": Kernel(35, 35, defocus=defocus_flag, combo=combo_flag),
                     "combo CT focus": Kernel(35, 35, conjuncture=conjuncture_flag, combo=combo_flag),
                     "combo CT defocus": Kernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag,
                                                    combo=combo_flag)}
    end = time.time()
    print(end-start)
    # opt_kernels_gt = {"focus": DumpKernel(35, 35), "defocus": DumpKernel(35, 35, defocus=defocus_flag),
    #                "CT focus": DumpKernel(35, 35, conjuncture=conjuncture_flag),
    #                "CT defocus": DumpKernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag)}
    #
    # combo_kernels_gt = {"combo focus": DumpKernel(35, 35, combo=combo_flag),
    #                  "combo defocus": DumpKernel(35, 35, defocus=defocus_flag, combo=combo_flag),
    #                  "combo CT focus": DumpKernel(35, 35, conjuncture=conjuncture_flag, combo=combo_flag),
    #                  "combo CT defocus": DumpKernel(35, 35, defocus=defocus_flag, conjuncture=conjuncture_flag,
    #                                             combo=combo_flag)}
    #
    # for k, v in opt_kernels.items():
    #     print(v.kernels.equal(opt_kernels_gt[k].kernels))
    #     print(v.scales.equal(opt_kernels_gt[k].scales))
    #
    # for k, v in combo_kernels.items():
    #     print(v.kernels.equal(combo_kernels_gt[k].kernels))
    #     print(v.scales.equal(combo_kernels_gt[k].scales))