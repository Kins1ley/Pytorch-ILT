def check_equal_image(cpp_file, py_image):
    '''
    check if the result of the python image is equal to the cpp iamge
    :param cpp_file: the position index of the pixels of cpp image that are not 0, represented as:
        1368 1048
        1368 1049
        1368 1050
        1368 1051
        1368 1052
    :param py_image: the tensor of one channel of the python output image
    :return: None
    '''
    with open(cpp_file) as f:
        cpp_outer_image = f.readlines()
    for i in range(len(cpp_outer_image)):
        cpp_outer_image[i] = cpp_outer_image[i].split(" ")
        cpp_outer_image[i][1] = cpp_outer_image[i][1][:-1]

    for i in range(len(cpp_outer_image)):
        for j in range(2):
            cpp_outer_image[i][0] = int(cpp_outer_image[i][0])
            cpp_outer_image[i][1] = int(cpp_outer_image[i][1])

    print(py_image[cpp_outer_image[0][0], cpp_outer_image[0][1]])
    for pos in cpp_outer_image:
        if py_image[pos[0], pos[1]] == 0:
            print("error")
