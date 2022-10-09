with open("epe_sample_py.txt") as f:
    py = f.readlines()

with open("epe_sample_cpp.txt") as f:
    cpp = f.readlines()

print(cpp)
print(py)
print(cpp==py)