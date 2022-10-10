import torch


ZERO_ERROR = 0.000001


def unit_test_params(py_data, cpp_data_file, size_x, size_y):

    with open(cpp_data_file) as f:
        cpp_params_result = f.readlines()

    cpp_params_result = cpp_params_result[0].split(" ")[:-1]
    for i in range(len(cpp_params_result)):
        cpp_params_result[i] = float(cpp_params_result[i])

    cpp_params_result = torch.tensor(cpp_params_result, dtype=torch.float64).reshape(size_x, size_y)
    # print((torch.abs((py_data - cpp_params_result)) < ZERO_ERROR).all())
    torch.set_printoptions(precision=12)
    # for i in range(2048):
    #     if cpp_params_result[1080, i] != py_data[1080, i]:
    #         print(cpp_params_result[1080,i], py_data[1080,i])
