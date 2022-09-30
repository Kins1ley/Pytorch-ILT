import struct
import torch

# def hex2int(i0, i1, i2, i3):
#     # print(int(i0), int(i1), int(i2), int(i3))
#     return int(i3) + int(i2) * (16 ** 2) + int(i1) * (16 ** 4) + int(i0) * (16 ** 6)

class Kernel:

    def __init__(self, knx, kny, flag_defocus):
        self.knx = knx
        self.kny = kny
        self.knum = 0
        self.flag_defocus = flag_defocus
        self.scales = self.read_scales()
        self.kernels = self.read_kernels()

    def read_scales(self):
        if not self.flag_defocus:
            scale_file = "cuilt/Kernels/M1OPC/scales.txt"
        else:
            scale_file = "cuilt/Kernels/M1OPC_def/scales.txt"
        with open(scale_file, "r") as f:
            scales_content = f.readlines()
        scales = [float(scale[:-1]) for scale in scales_content[1:]]
        print(scales)
        self.knum = int(scales_content[0][:-1])
        temp = torch.tensor(scales, dtype=torch.float64)
        print(temp)
        return temp

    def read_kernels(self):
        kernels = []
        for i in range(self.knum):
            kernel = []
            if not self.flag_defocus:
                binary_file = "cuilt/Kernels/M1OPC/fh" + str(i) + ".bin"
            else:
                binary_file = "cuilt/Kernels/M1OPC_def/fh" + str(i) + ".bin"
            file = open(binary_file, "rb")
            data = file.read()
            for j in range(5, len(data)//4-1):
                values = (data[4*j+3], data[4*j+2], data[4*j+1], data[4*j])
                encode_data = struct.pack("4B", *values)
                decode_data = struct.unpack('f', encode_data)
                kernel.append(decode_data)
            kernels.append(kernel)
        # print(kernels[15][1258], kernels[15][1259])

        return torch.tensor(kernels, dtype=torch.float64)

if __name__ == "__main__":
    kernel_focus = Kernel(35, 35, 0)
