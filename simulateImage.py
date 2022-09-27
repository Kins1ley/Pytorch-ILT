import torch

imx = 2048
imy = 2048
kernel_x = 35
kernel_y = 35
kernel_num = 24

def compute_image(m_cmask):
    pass

def shift(m_cmask):
    shift_cmask = torch.zeros(m_cmask.size(), dtype=torch.complex64)
    shift_cmask[:imx//2, :imy//2] = m_cmask[imx//2:, imy//2:]     # 1 = 4
    shift_cmask[:imx//2, imy//2:] = m_cmask[imx//2:, :imy//2]     # 2 = 3
    shift_cmask[imx//2:, :imy//2] = m_cmask[:imx//2, imy//2:]     # 3 = 2
    shift_cmask[imx//2:, imy//2:] = m_cmask[:imx//2, :imy//2]     # 4 = 1
    return shift_cmask

def kernel_mult(knx, kny, kernel, mask_fft):
    imxh = imx//2
    imxy = imy//2
    xoff = imxh - knx//2
    yoff = imxy - kny//2
    output = torch.zeros(mask_fft.size(), dtype=torch.complex64)
    output[xoff:xoff+knx, yoff:yoff+kny] = torch.mul(kernel, mask_fft[xoff:xoff+knx, yoff:yoff+kny])
    return output

if __name__ == "__main__":
    # background_real = 1.0
    # background_img = 0.0
    dose = 1.0
    scale = torch.randn((kernel_num, ), dtype=torch.float64)
    print(scale.size())
    kernel = torch.zeros((kernel_x, kernel_y, kernel_num), dtype=torch.complex64)
    m_mask = torch.randn((imx, imy), dtype=torch.complex64)
    # print(m_mask)
    # m_cmask = torch.zeros((imx, imy))
    # m_cmask[:, :] = m_mask
    m_mask = shift(m_mask)
    print(m_mask.size())
    m_fftmask = torch.fft.fft2(m_mask, norm="forward")
    m_fftmask = shift(m_fftmask)
    mul_fft = torch.zeros(m_fftmask.size(), dtype=torch.complex64)
    for i in range(kernel_num):
        # temp = kernel_mult(kernel_x, kernel_y, kernel[:, :, i], m_fftmask)
        temp = shift(kernel_mult(kernel_x, kernel_y, kernel[:, :, i], m_fftmask))
        # temp = torch.fft.ifft2(temp)
        temp = shift(torch.fft.ifft2(temp))
        print(torch.abs(temp).size())
        mul_fft += scale[i] * torch.pow(torch.abs(temp), 2)
        # print(torch.pow(torch.abs(temp), 2).size())
        # break

    # m_mask = torch.randn((imx, imy))
    # m_fftmask = torch.fft.rfft2(m_mask, norm="forward")
