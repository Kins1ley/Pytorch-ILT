import torch

image = torch.tensor([[1,2,1], [0, 1, 0], [2, 0, 0]])

image_fft = torch.fft.fft2(image)

print(image)

print("after fft\n")
print(image_fft)