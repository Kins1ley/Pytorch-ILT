import torch
import time
from constant import device
from utils import unit_test_kernel, unit_test_mask_fft

imx = 2048
imy = 2048
kernel_x = 35
kernel_y = 35
# kernel_num = 24


def shift(cmask):
    shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex128, device=device)
    shift_cmask[:imx//2, :imy//2] = cmask[imx//2:, imy//2:]     # 1 = 4
    shift_cmask[:imx//2, imy//2:] = cmask[imx//2:, :imy//2]     # 2 = 3
    shift_cmask[imx//2:, :imy//2] = cmask[:imx//2, imy//2:]     # 3 = 2
    shift_cmask[imx//2:, imy//2:] = cmask[:imx//2, :imy//2]     # 4 = 1
    return shift_cmask


def kernel_mult(knx, kny, kernel, mask_fft):
    imxh = imx//2
    imxy = imy//2
    xoff = imxh - knx//2
    yoff = imxy - kny//2
    output = torch.zeros(mask_fft.size(), dtype=torch.complex128, device=device)
    output[xoff:xoff+knx, yoff:yoff+kny] = torch.mul(kernel, mask_fft[xoff:xoff+knx, yoff:yoff+kny])
    return output


# def expand_kernel(knx, kny, kernel, kernel_level):
#     imxh = imx//2
#     imxy = imy//2
#     xoff = imxh - knx//2
#     yoff = imxy - kny//2
#     kernel_expand = torch.zeros((imx, imy, kernel_level), dtype=torch.complex128, device=device)
#     kernel_expand[xoff:xoff+knx, yoff:yoff+kny, :] = kernel
#     return kernel_expand


# def compute_image1(cmask, kernel, scale, workx, worky, dose, kernel_level):
#     # 优化版本，但是时间更久
#     kernel_num = kernel_level
#     cmask = shift(cmask)
#     cmask_fft = torch.fft.fft2(cmask, norm="forward")
#     cmask_fft = shift(cmask_fft)
#
#     cmask_fft_expand = cmask_fft.unsqueeze(-1)
#     # print(cmask_fft_expand.size())
#     # mul_fft = torch.zeros(cmask_fft.size(), dtype=torch.complex64)
#     kernel_expand = expand_kernel(kernel_x, kernel_y, kernel, kernel_level)
#     mul_fft = torch.mul(cmask_fft_expand, kernel_expand)
#     mul_fft = shift(mul_fft)
#     mul_ifft = shift(torch.fft.ifft2(mul_fft, dim=(-3, -2), norm="forward"))
#     scale_expand = scale.unsqueeze(-1).unsqueeze(-1).reshape(1, 1, -1)
#     mul_ifft = mul_ifft * scale_expand
#     result = torch.sum(torch.abs(mul_ifft) ** 2, dim=-1)
#     # print(result.size())
#     # print(mul_ifft.size())
#     # for i in range(kernel_num):
#     #     temp = shift(kernel_mult(kernel_x, kernel_y, kernel[:, :, i], cmask_fft))
#     #     # temp = torch.fft.ifft2(temp)
#     #     temp = shift(torch.fft.ifft2(temp, norm="forward"))
#     #     print(torch.abs(temp).size())
#     #     mul_fft += scale[i] * torch.pow(torch.abs(temp), 2)
#     return result


def compute_image(cmask, kernel, scale, workx, worky, dose, kernel_level):
    kernel_num = kernel_level
    cmask = shift(cmask)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    cmask_fft = shift(cmask_fft)
    # unit_test_mask_fft(cmask_fft, "/Users/zhubinwu/research/opc-hsd/cuilt/build/init_fft_cmask.txt")
    mul_fft = torch.zeros(cmask_fft.size(), dtype=torch.float64, device=device)
    print(kernel_num)
    torch.set_printoptions(precision=8)
    for i in range(kernel_num):
        temp = kernel_mult(kernel_x, kernel_y, kernel[:, :, i], cmask_fft)
        if i == 0:
            print(temp[1024, 1024])
        temp = shift(temp)
        if i == 0:
            print(temp[0, 0])
        # temp = torch.fft.ifft2(temp)
        temp = torch.fft.ifft2(temp, norm="forward")
        if i == 0:
            print(temp[0, 0])
        temp = shift(temp)
        # print(torch.abs(temp).size())
        mul_fft += scale[i] * torch.pow(torch.abs(temp), 2)
        if i == 0:
            print(temp[1024, 1024])
    return mul_fft


# 用dose作用mask
def mask_float(mask, dose):
    return (dose * mask).to(torch.complex128)


def simulate_image(mask, kernel, scale, dose, kernel_level):
    # if kernel_type == 0:
    #     kernel = torch.randn((kernel_x, kernel_y, kernel_num), dtype=torch.complex64)
    #     scale = torch.randn((kernel_num,), dtype=torch.float64)
    cmask = mask_float(mask, dose)

    # pixel_positive_1 = 0
    # pixel_negative_1 = 0
    # pixel_0 = 0
    # for i in range(2048):
    #     for j in range(2048):
    #         if cmask[i, j] == cmask[548, 1224]:
    #             pixel_negative_1 += 1
    #         elif cmask[i, j] == cmask[613, 1250]:
    #             pixel_positive_1 += 1
    # print(pixel_negative_1, pixel_positive_1) # 1411056 227344
    image = compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)
    return image

# if __name__ == "__main__":
#     kernel_num = 15
#     mask = torch.randn((imx, imy), dtype=torch.float64)
#     kernel = torch.randn((kernel_x, kernel_y, kernel_num), dtype=torch.complex128)
#     scale = torch.randn((kernel_num,), dtype=torch.float64)
#     dose = 1.0
#     kernel_level = 15   #kernel数目
#     start = time.time()
#     output = simulate_image(mask, kernel, scale, dose, kernel_level)
#     end = time.time()
#     print(end-start)
    # print(output)
    # print((output1 == output2).all())
    # print(output)
