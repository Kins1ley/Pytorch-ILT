import torch


ZERO_ERROR = 0.000001

def unit_test_mask_fft(py_data, cpp_data_file):

    with open(cpp_data_file) as f:
        cpp_params_result = f.readlines()

    cpp_params_result = cpp_params_result[0].split(" ")[:-1]
    for i in range(len(cpp_params_result)):
        cpp_params_result[i] = float(cpp_params_result[i])

    cpp_params_result = torch.tensor(cpp_params_result)

    for i in range(2048):
        for j in range(2048):
            if torch.abs(py_data[i, j].real - cpp_params_result[i * 2048 * 2 + j * 2]) > ZERO_ERROR:
                print("real value error")
                print(py_data[i, j].real)
                print(cpp_params_result[i * 2048 * 2 + j * 2])
            if torch.abs(py_data[i, j].imag - cpp_params_result[i * 2048 * 2 + j * 2 + 1]) > ZERO_ERROR:
                print("imag value error")
                print(py_data[i, j].img)
                print(cpp_params_result[i * 2048 * 2 + j * 2 + 1])

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    with open("/Users/zhubinwu/research/opc-hsd/cuilt/build/init_fft_cmask.txt") as f:
        nominal_dose_kernel = f.readlines()
    nominal_dose_kernel = nominal_dose_kernel[0].split(" ")[:-1]
    for i in range(len(nominal_dose_kernel)):
        nominal_dose_kernel[i] = float(nominal_dose_kernel[i])
    # # print(nominal_dose_kernel[-34])
    nominal_dose_kernel = torch.tensor(nominal_dose_kernel)
    print(len(nominal_dose_kernel))
    print(nominal_dose_kernel[2048*2*1024+1024*2])
    print(nominal_dose_kernel[2048 * 2 * 1024 + 1024 * 2 + 1])

