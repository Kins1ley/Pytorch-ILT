import struct


# def hex2int(i0, i1, i2, i3):
#     # print(int(i0), int(i1), int(i2), int(i3))
#     return int(i3) + int(i2) * (16 ** 2) + int(i1) * (16 ** 4) + int(i0) * (16 ** 6)

def read_kernel(flag_defocus):
    if not flag_defocus:
        fname = "cuilt/Kernels/M1OPC/scales.txt"
    else:
        fname = "cuilt/Kernels/M1OPC_def/scales.txt"
    i = 0
    with open(fname, "r") as f:
        a = f.readlines()
    scales = a[1:]
    for i in range(len(scales)):
        scales[i] = float(scales[i][:-1])
    nk = int(a[0][:-1])
    kernels = []
    for i in range(nk):
        kernel = []
        if not flag_defocus:
            fname = "cuilt/Kernels/M1OPC/fh" + str(i) + ".bin"
        else:
            fname = "cuilt/Kernels/M1OPC_def/fh" + str(i) + ".bin"
        file = open(fname,"rb")
        data = file.read()
        # print(len(data[24:])//4)
        # # file.seek(0, 0)
        # print(data[:4])
        # values = (data[3], data[2], data[1], data[0])
        # packed_data = struct.pack("4B", *values)
        # print(packed_data)
        # data_raw = struct.unpack('i', packed_data)
        # print(data_raw)
        # break
        print(len(data))
        for j in range(5, len(data)//4-1):
            values = (data[4*j+3], data[4*j+2], data[4*j+1], data[4*j])
            encode_data = struct.pack("4B", *values)
            # print(packed_data)
            decode_data = struct.unpack('f', encode_data)
            kernel.append(decode_data)
        # print("*******kernel[1000], kernel[1001]*********")
        # print(kernel[2450*15+1258], kernel[2450*15+1259])
        kernels.append(kernel)
    print(len(kernels))
    print(kernels[15][1258], kernels[15][1259])
if __name__ == "__main__":
    read_kernel(0)