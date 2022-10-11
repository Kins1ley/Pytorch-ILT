import torch
import time
# from constant import device
from simulate_image import simulate_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imx = 2048
imy = 2048
kernel_x = 35
kernel_y = 35


def shift2(cmask):
    shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex128, device=device)
    shift_cmask[:, :imx//2, :imy//2] = cmask[:, imx//2:, imy//2:]     # 1 = 4
    shift_cmask[:, :imx//2, imy//2:] = cmask[:, imx//2:, :imy//2]     # 2 = 3
    shift_cmask[:, imx//2:, :imy//2] = cmask[:, :imx//2, imy//2:]     # 3 = 2
    shift_cmask[:, imx//2:, imy//2:] = cmask[:, :imx//2, :imy//2]     # 4 = 1
    return shift_cmask


def kernel_mult2(knx, kny, kernel, mask_fft, kernel_num):
    imxh = imx//2
    imxy = imy//2
    xoff = imxh - knx//2
    yoff = imxy - kny//2
    kernel = kernel.permute(2, 0, 1)
    temp_mask_fft = mask_fft[0, xoff:xoff+knx, yoff:yoff+kny]
    output = torch.zeros([kernel_num, imx, imy], dtype=torch.complex128, device=device)
    output[:, xoff:xoff+knx, yoff:yoff+kny] = temp_mask_fft * kernel[:kernel_num, :, :]
    return output


def compute_image2(cmask, kernel, scale, workx, worky, dose, kernel_level):
    kernel_num = kernel_level
    cmask = torch.unsqueeze(cmask, 0)
    cmask = shift2(cmask)
    cmask_fft = torch.fft.fft2(cmask)
    cmask_fft = shift2(cmask_fft)
    temp = shift2(kernel_mult2(kernel_x, kernel_y, kernel, cmask_fft, kernel_num))
    temp = shift2(torch.fft.ifft2(temp))
    scale = scale[:kernel_num].unsqueeze(1).unsqueeze(2)
    mul_fft = torch.sum(scale * torch.pow(torch.abs(temp), 2), dim=0).to(device)
    return mul_fft


def mask_float2(mask, dose):
    return (dose * mask).to(torch.complex128)


def simulate_image2(mask, kernel, scale, dose, kernel_level):
    cmask = mask_float2(mask, dose)
    image = compute_image2(cmask, kernel, scale, 0, 0, dose, kernel_level)
    return image
    # print("image is equal to image2 or not", image.equal(image2))
    # return image

if __name__ == "__main__":
    # test the correctness
    kernel_num = 15
    mask = torch.randn((imx, imy), dtype=torch.float64).to(device)
    kernel = torch.randn((kernel_x, kernel_y, kernel_num), dtype=torch.complex128).to(device)
    scale = torch.randn((kernel_num,), dtype=torch.float64).to(device)
    dose = 1.0
    kernel_level = 15   #kernel数目
    # start = time.time()
    # for i in range(100):
    output_test = simulate_image2(mask, kernel, scale, dose, kernel_level)
    output_gt = simulate_image(mask, kernel, scale, dose, kernel_level)
    # end = time.time()
    # print(end-start)
    print(output_gt.equal(output_test))

    #profiling
