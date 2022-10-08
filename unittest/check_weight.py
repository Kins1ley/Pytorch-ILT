with open("epe_weight_py.txt") as f:
    py = f.readlines()

for i in range(len(py)):
    if py[i][-1] == "\n":
        py[i] = py[i][:-1]
# print(len(py))

with open("epe_weight_cpp.txt") as f:
    cpp = f.readlines()

for i in range(len(py)):
    if cpp[i][-1] == "\n":
        cpp[i] = cpp[i][:-1]

for i in range(len(cpp)):
    if cpp[i] != py[i]:
        print(i)