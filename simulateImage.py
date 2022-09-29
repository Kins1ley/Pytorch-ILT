import torch

imx = 2048
imy = 2048
kernel_x = 35
kernel_y = 35
# kernel_num = 24


def shift(cmask):
    shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex64)
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
    output = torch.zeros(mask_fft.size(), dtype=torch.complex64)
    output[xoff:xoff+knx, yoff:yoff+kny] = torch.mul(kernel, mask_fft[xoff:xoff+knx, yoff:yoff+kny])
    return output


def compute_image(cmask, kernel, scale, workx, worky, dose, kernel_level):
    kernel_num = kernel_level
    cmask = shift(cmask)
    cmask_fft = torch.fft.fft2(cmask)
    cmask_fft = shift(cmask_fft)
    mul_fft = torch.zeros(cmask_fft.size(), dtype=torch.float64)
    for i in range(kernel_num):
        temp = shift(kernel_mult(kernel_x, kernel_y, kernel[:, :, i], cmask_fft))
        # temp = torch.fft.ifft2(temp)
        temp = shift(torch.fft.ifft2(temp))
        print(torch.abs(temp).size())
        mul_fft += scale[i] * torch.pow(torch.abs(temp), 2)
    return mul_fft


def mask_float(mask, dose):
    return (dose * mask).to(torch.complex64)


def simulate_image(mask, kernel_type, dose, kernel_level):
    kernel_num = kernel_level
    if kernel_type == 0:
        kernel = torch.randn((kernel_x, kernel_y, kernel_num), dtype=torch.complex64)
        scale = torch.randn((kernel_num,), dtype=torch.float64)
    cmask = mask_float(mask, dose)
    image = compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)

    return image


if __name__ == "__main__":
    mask = torch.randn((imx, imy), dtype=torch.float64)
    # kernel_types = [""]
    dose = 1.0
    kernel_level = 15
    output = simulate_image(mask, 0, dose, kernel_level)
    print(output)
