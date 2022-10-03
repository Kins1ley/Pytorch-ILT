import struct
import torch


class Kernel(object):

    def __init__(self, knx, kny, device, defocus=0, conjuncture=0, combo=0):
        self.knx = knx
        self.kny = kny
        # self.device = device
        self.flag_defocus = defocus
        self.conjuncture = conjuncture
        self.combo = combo
        # other kernel-related variables
        self.knum = 0  # kernel number, if combo = 0, knum = 24, else, knum = 1
        self.combo_num = 0  # the number of combined kernels, if combo = 0, combo_num = 0, else, combo_num = 9
        self.scales = self.read_scales()

        if not self.conjuncture:
            self.kernels = self.read_kernels().to(device)
        else:
            self.kernels = torch.conj(self.read_kernels().transpose(0, 1)).to(device)

        if self.combo:
            self.combo_kernel()

    def read_scales(self):
        if not self.flag_defocus:
            scale_file = "./kernels/M1OPC/scales.txt"
        else:
            scale_file = "./kernels/M1OPC_def/scales.txt"
        with open(scale_file, "r") as f:
            scales_content = f.readlines()
        scales = [float(scale[:-1]) for scale in scales_content[1:]]
        # print(scales)
        self.knum = int(scales_content[0][:-1])
        temp = torch.tensor(scales, dtype=torch.float64)
        # print(temp)
        return temp

    def byte2float(self, bytes):
        encode_data = struct.pack("4B", *bytes)     # 按照unsigned char*打包四个字节
        decode_data = struct.unpack('f', encode_data)
        return decode_data

    def read_kernels(self):
        kernels = []
        for i in range(self.knum):
            kernel = []
            if not self.flag_defocus:
                binary_file = "./kernels/M1OPC/fh" + str(i) + ".bin"
            else:
                binary_file = "./kernels/M1OPC_def/fh" + str(i) + ".bin"
            file = open(binary_file, "rb")
            data = file.read()
            # print(len(data)//4)
            for j in range(5, len(data)//4-1, 2):
                real_values = self.byte2float((data[4*j+3], data[4*j+2], data[4*j+1], data[4*j]))   # 逆排序是为了实现高位寻址->低位寻址
                imag_values = self.byte2float((data[4*j+7], data[4*j+6], data[4*j+5], data[4*j+4]))
                kernel.append(torch.tensor([real_values[0] + 1j*imag_values[0]], dtype=torch.complex64))
            kernels.append(kernel)
        kernels = torch.tensor(kernels, dtype=torch.complex64)
        kernels = (kernels.view((24, 35, 35))).permute(1, 2, 0)
        return kernels

    def combo_kernel(self):
        self.knum = 1
        self.combo_num = 9
        scales = torch.sqrt(self.scales)
        self.scales = torch.tensor([1], dtype=torch.float64)
        combo_kernel = torch.zeros([self.knx, self.kny], dtype=torch.complex64).to(device)
        for i in range(self.combo_num):
            combo_kernel += scales[i] * self.kernels[:, :, i]
        self.kernels = combo_kernel


if __name__ == "__main__":
    # import os
    # pwd = os.getcwd()
    # print(pwd)
    torch.set_printoptions(precision=6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    defocus_flag = 1
    conjuncture_flag = 1
    combo_flag = 1
    opt_kernels = {"focus": Kernel(35, 35, device), "defocus": Kernel(35, 35, device, defocus=defocus_flag),
                  "CT focus": Kernel(35, 35, device, conjuncture=conjuncture_flag),
                  "CT defocus": Kernel(35, 35, device, defocus=defocus_flag, conjuncture=conjuncture_flag)}

    combo_kernels = {"combo focus": Kernel(35, 35, device, combo=combo_flag),
                     "combo defocus": Kernel(35, 35, device, defocus=defocus_flag, combo=combo_flag),
                     "combo CT focus": Kernel(35, 35, device, conjuncture=conjuncture_flag, combo=combo_flag),
                     "combo CT defocus": Kernel(35, 35, device, defocus=defocus_flag, conjuncture=conjuncture_flag, combo=combo_flag)}
    # 已经测试了如下kernel数值和c++版本的都对的上
    kernel_focus = opt_kernels["focus"]
    kernels = kernel_focus.kernels
    print(kernels[13, 12, 0])
    print(kernels[13, 12, 1])
    print(kernels[13, 12, 2])
    print(kernels[13, 12, 3])
    print(kernels[13, 12, 4])
    print(kernels[13, 12, 5])
    print(kernels[13, 12, 23])
    # combo_kernel_focus = combo_kernels["combo CT defocus"]
    # kernels = combo_kernel_focus.kernels
    # print(kernels.size())
    # print(kernels[13, 12])
    # print(kernels[13, 15])
    # print(kernels[14, 13])
    # print(kernels[16, 17])
    # print(kernels.device)

