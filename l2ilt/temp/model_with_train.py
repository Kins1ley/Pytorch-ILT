import torch
import sys
import time
import torch.nn as nn
sys.path.append("..")
from constant import *
from dataset import ICCADTrain
from kernels import Kernel
from utils import write_image_file
import matplotlib.pyplot as plt


class Simulator(nn.Module):
    def __init__(self):
        super(Simulator, self).__init__()
        self.imx = OPC_TILE_X
        self.imy = OPC_TILE_Y
        self.kernel_x = KERNEL_X
        self.kernel_y = KERNEL_Y

    def shift(self, cmask):
        shift_cmask = torch.zeros(cmask.size(), dtype=torch.complex64, device=device)
        shift_cmask[:, :, :self.imx // 2, :self.imy // 2] = cmask[:, :, self.imx // 2:, self.imy // 2:]  # 1 = 4
        shift_cmask[:, :, :self.imx // 2, self.imy // 2:] = cmask[:, :, self.imx // 2:, :self.imy // 2]  # 2 = 3
        shift_cmask[:, :, self.imx // 2:, :self.imy // 2] = cmask[:, :, :self.imx // 2, self.imy // 2:]  # 3 = 2
        shift_cmask[:, :, self.imx // 2:, self.imy // 2:] = cmask[:, :, :self.imx // 2, :self.imy // 2]  # 4 = 1
        return shift_cmask

    def kernel_mult(self, knx, kny, kernel, mask_fft, kernel_num):
        imxh = self.imx // 2
        imxy = self.imy // 2
        xoff = imxh - knx // 2
        yoff = imxy - kny // 2
        kernel = kernel.permute(2, 0, 1)
        temp_mask_fft = mask_fft[:, :, xoff:xoff + knx, yoff:yoff + kny]
        output = torch.zeros([1, kernel_num, self.imx, self.imy], dtype=torch.complex64, device=device)
        output[0, :, xoff:xoff + knx, yoff:yoff + kny] = temp_mask_fft * kernel[:kernel_num, :, :]
        return output

    def compute_image(self, cmask, kernel, scale, workx, worky, dose, kernel_level):
        kernel_num = kernel_level
        # cmask = torch.unsqueeze(cmask, 0)
        cmask = self.shift(cmask)
        cmask_fft = torch.fft.fft2(cmask, norm="forward")
        cmask_fft = self.shift(cmask_fft)
        temp = self.shift(self.kernel_mult(self.kernel_x, self.kernel_y, kernel, cmask_fft, kernel_num))
        temp = self.shift(torch.fft.ifft2(temp, norm="forward"))
        if kernel_level == 1:
            return temp[0]
        elif kernel_level == 15 or kernel_level == 24:
            scale = scale[:kernel_num]
            # print(scale.size())
            mul_fft = torch.sum(scale * torch.pow(torch.abs(temp), 2), dim=1, keepdim=True).to(device)
            return mul_fft

    def mask_float(self, mask, dose):
        return (dose * mask).to(torch.complex64)

    def forward(self, mask, kernel, scale, dose, kernel_level):
        # mask [N, C=1, H, W], kernel [num_kernel, H, W]
        cmask = self.mask_float(mask, dose)
        image = self.compute_image(cmask, kernel, scale, 0, 0, dose, kernel_level)
        return image


class GradientBlock(nn.Module):
    def __init__(self):
        super(GradientBlock, self).__init__()
        self.theta_z = PHOTORISIST_SIGMOID_STEEPNESS
        self.target_intensity = TARGET_INTENSITY
        self.theta_m = MASKRELAX_SIGMOID_STEEPNESS
        self.gamma = 4
        self.conv0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.filter = torch.zeros([1, OPC_TILE_Y, OPC_TILE_X], dtype=torch.float32).to(device)
        self.filter[0, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1.0
        self.update_mask = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.step_size = torch.ones([1, OPC_TILE_Y, OPC_TILE_X], dtype=torch.float32, device=device) * 0.5 * self.filter
        self.simulator = Simulator()

    def forward(self, param, z_target):
        mask = self.update_mask(self.theta_m * param) * self.filter
        intensity = self.conv0(mask)
        # print(intensity.size())
        z_norm = self.sigmoid(self.theta_z * (intensity - self.target_intensity))
        # print(z_norm.size())
        # print(z_target.size())
        term1 = (z_norm - z_target) ** 3 * z_norm * (1-z_norm) * self.conv1(mask)
        term2 = self.conv2(term1)
        term3 = (z_norm - z_target) ** 3 * z_norm * (1-z_norm) * self.conv3(mask)
        term4 = self.conv4(term3)
        gradient = self.gamma * self.theta_z * self.theta_m * (term2 + term4) * mask * (1-mask)
        # print(gradient[0,0,1024,1020:1030])
        update_param = param - self.step_size * gradient
        # update_param = update_param * self.filter
        return update_param


class L2ILT(nn.Module):
    def __init__(self, kernels, scales):
        super(L2ILT, self).__init__()
        self.kernels = kernels
        self.scales = scales
        self.block1 = GradientBlock()
        self.block2 = GradientBlock()
        self.block3 = GradientBlock()
        self.block4 = GradientBlock()
        self.block5 = GradientBlock()
        self.update_mask = nn.Sigmoid()
        self.intensity2znorm = nn.Sigmoid()
        self.simulator = Simulator()
        self.filter = torch.zeros([1, OPC_TILE_Y, OPC_TILE_X], dtype=torch.float32).to(device)
        self.filter[0, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1.0

    def forward(self, param, target):
        param_iter1 = self.block1(param, target)
        mask_iter1 = torch.sigmoid(MASKRELAX_SIGMOID_STEEPNESS * param_iter1) * self.filter
        intensity_iter1 = self.simulator(mask_iter1, self.kernels, self.scales, NOMINAL_DOSE, 24)
        znorm_iter1 = self.intensity2znorm(PHOTORISIST_SIGMOID_STEEPNESS * (intensity_iter1 - TARGET_INTENSITY))

        # param_iter2 = self.block2(param_iter1, target)
        # mask_iter2 = torch.sigmoid(MASKRELAX_SIGMOID_STEEPNESS * param_iter2) * self.filter
        # intensity_iter2 = self.simulator(mask_iter2, self.kernels, self.scales, MAX_DOSE, 24)
        # znorm_iter2 = self.intensity2znorm(PHOTORISIST_SIGMOID_STEEPNESS * (intensity_iter2 - TARGET_INTENSITY))
        #
        # param_iter3 = self.block3(param_iter2, target)
        # mask_iter3 = torch.sigmoid(MASKRELAX_SIGMOID_STEEPNESS * param_iter3) * self.filter
        # intensity_iter3 = self.simulator(mask_iter3, self.kernels, self.scales, MAX_DOSE, 24)
        # znorm_iter3 = self.intensity2znorm(PHOTORISIST_SIGMOID_STEEPNESS * (intensity_iter3 - TARGET_INTENSITY))
        #
        # param_iter4 = self.block4(param_iter3, target)
        # mask_iter4 = torch.sigmoid(MASKRELAX_SIGMOID_STEEPNESS * param_iter4) * self.filter
        # intensity_iter4 = self.simulator(mask_iter4, self.kernels, self.scales, MAX_DOSE, 24)
        # znorm_iter4 = self.intensity2znorm(PHOTORISIST_SIGMOID_STEEPNESS * (intensity_iter4 - TARGET_INTENSITY))
        #
        # param_iter5 = self.block5(param_iter4, target)
        # mask_iter5 = torch.sigmoid(MASKRELAX_SIGMOID_STEEPNESS * param_iter5) * self.filter
        # intensity_iter5 = self.simulator(mask_iter5, self.kernels, self.scales, MAX_DOSE, 24)
        # znorm_iter5 = self.intensity2znorm(PHOTORISIST_SIGMOID_STEEPNESS * (intensity_iter5 - TARGET_INTENSITY))
        return mask_iter1, znorm_iter1
            # , znorm_iter2, znorm_iter3, znorm_iter4, znorm_iter5


if __name__ == "__main__":
    batch_size = 1

    filter = torch.zeros([1, OPC_TILE_Y, OPC_TILE_X], dtype=torch.float32).to(device)
    filter[0, LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X] = 1.0

    train_dataset = ICCADTrain("ICCAD2013/")
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=1,
                                              shuffle=False)

    kernels = {"focus": Kernel(35, 35), "defocus": Kernel(35, 35, defocus=1),
                    "combo focus": Kernel(35, 35, combo=1),
                    "combo defocus": Kernel(35, 35, defocus=1, combo=1),
                    "combo CT focus": Kernel(35, 35, conjuncture=1, combo=1),
                    "combo CT defocus": Kernel(35, 35, defocus=1, conjuncture=1, combo=1)}
    model = L2ILT(kernels["focus"].kernels, kernels["focus"].scales).to(device)
    simulator = Simulator()
    # focus_kernels = kernels["focus"].kernels
    # focus_scales = kernels["focus"].scales
    # criterion = OPCLoss(focus_kernels, focus_scales)
    criterion = nn.MSELoss(reduction="sum")
    num_epochs = 100
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_dataset)
    # for parameters in model.parameters():
    #     print(parameters.grad)

    for epoch in range(num_epochs):
        i = 0
        for data_pair in train_dataloader:
            i += 1
            param = data_pair["param"].to(device)
            target = data_pair["target"].to(device)
            # start = time.time()
            mask1, znorm1 = model(param, target)
            # end = time.time()
            # print(end-start)
            loss = criterion(znorm1, target)
                   # + criterion(output2, target) + criterion(output3, target) + criterion(output4, target) + criterion(output5, target)
            optimizer.zero_grad()
            loss.backward()
            # for parameters in model.parameters():
            #     print(parameters.grad)
            optimizer.step()
            if epoch % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i, total_step, loss.item()))
                bmask = torch.zeros(mask1.size(), dtype=torch.float32)
                bmask[mask1 >= MASK_PRINTABLE_THRESHOLD] = 1
                image = simulator(bmask, kernels["focus"].kernels, kernels["focus"].scales, NOMINAL_DOSE, 24)
                plt.imshow(write_image_file(image[0,0], NOMINAL_DOSE))
                plt.savefig("./save_images/mask" + str(i) + "epoch_" + str(epoch) + ".png")
                plt.clf()
                # print('last item direct loss {}'.format(criterion(output1, target)))


    # data_pair = train_dataset[0]
    # param = data_pair["param"].to(device).unsqueeze(0)
    # target = data_pair["target"].to(device).unsqueeze(0)
    # output1, output2, output3 = model(param, target)
    # loss = criterion(output1, target) + criterion(output2, target) + criterion(output3, target)
    # loss.backward()
    # print(loss)
    # target = target[0]
    # loss = criterion(outputs, target)
    # loss.requires_grad_(True)
    # print(loss)
    # loss.backward()
    # # for parameter in model.parameters():
    # #     print(parameter)
    # for parameter in model.parameters():
    #     print(parameter.grad)
    # t = torch.tensor(outputs[0], requires_grad=True)
    # loss = criterion(t, target[0])
    # loss.backward()

    # for epoch in range(num_epochs):
    #     i = 0
    #     for data_pair in train_dataset:
    #         i += 1
    #         param = data_pair["param"].to(device)
    #         target = data_pair["target"].to(device)
    #         param = param.unsqueeze(0).unsqueeze(0)
    #         target = target.unsqueeze(0)
    #         start = time.time()
    #         outputs = model(param, target)
    #         end = time.time()
    #         # print("time", end-start)
    #         # print(target[0].dtype)
    #         # print(outputs[0].dtype)
    #         loss = criterion(outputs[-1], target[0])
    #         # print(outputs[0].size())
    #         # print(target[0].size())
    #                # criterion(outputs[3], target) + criterion(outputs[4], target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         # print(outputs[0].requires_grad)
    #         # print(outputs[0].grad)
    #         optimizer.step()
    #
    #         # if (epoch+1) % 5 == 0:
    #         print('Epoch [{}/{}], Step [{}/{}], Loss: {:.7f}'
    #               .format(epoch + 1, num_epochs, i, total_step, loss.item()))