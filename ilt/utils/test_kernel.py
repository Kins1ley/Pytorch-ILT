import torch


ZERO_ERROR = 0.000001

def unit_test_kernel(py_data, cpp_data_file):

    with open(cpp_data_file) as f:
        cpp_params_result = f.readlines()

    cpp_params_result = cpp_params_result[0].split(" ")[:-1]
    for i in range(len(cpp_params_result)):
        cpp_params_result[i] = float(cpp_params_result[i])

    cpp_params_result = torch.tensor(cpp_params_result, dtype=torch.float64)
    # print((torch.abs((py_data - cpp_params_result)) < ZERO_ERROR).all())
    torch.set_printoptions(precision=12)
    for k in range(24):
        for i in range(35):
            for j in range(35):
                if(torch.abs(py_data[i, j, k].real-cpp_params_result[k*35*35*2+i*35*2+j*2]) > ZERO_ERROR):
                    print(py_data[i, j, k].real)
                    print(cpp_params_result[k*35*35*2+i*35*2+j])
                    print("real value error")
                if (torch.abs(py_data[i, j, k].imag - cpp_params_result[k * 35 * 35 * 2 + i * 35 * 2 + j*2+1]) > ZERO_ERROR):
                    print(py_data[i, j, k].imag)
                    print(cpp_params_result[k * 35 * 35 * 2 + i * 35 * 2 + j+1])
                    print("imag value error")
# if __name__ == "__main__":
#     with open("/Users/zhubinwu/research/opc-hsd/cuilt/build/nominal_dose_kernel.txt") as f:
#         nominal_dose_kernel = f.readlines()
#     nominal_dose_kernel = nominal_dose_kernel[0].split(" ")[:-1]
#     print(nominal_dose_kernel[-34])
#     for i in range(len(nominal_dose_kernel)):
#         nominal_dose_kernel[i] = float(nominal_dose_kernel[i])
#     print(nominal_dose_kernel[-34])
#     print(len(nominal_dose_kernel))