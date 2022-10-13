import torch
import time

WEIGHT_REGULARIZATION = 0.5

def calculate_pixel_object_p8(diff_target, diff_image, discrete_penalty):
    # todo: test the correctness
    return torch.pow(diff_target, 4) + torch.pow(diff_image, 2) + WEIGHT_REGULARIZATION * discrete_penalty


diff_target = torch.randn([2048, 2048])
diff_image = torch.randn([2048, 2048])
discrete_penalty = torch.randn([2048, 2048])

start = time.time()
for i in range(100):
    a = torch.pow(diff_target, 4) + torch.pow(diff_image, 2) + WEIGHT_REGULARIZATION * discrete_penalty
end = time.time()
print(end-start)

start = time.time()
for i in range(100):
    a = calculate_pixel_object_p8(diff_target, diff_image, discrete_penalty)
end = time.time()
print(end-start)